"""AP544 diagnostic observer. No automatic installation; diagnostic bootstrap calls install()."""
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import types

INDEX = 'torch.ops.aten.index.Tensor'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical_method(node):
    # Device ordinal is the only rank-specific literal normalized.
    class Rank(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == 'get_raw_stream':
                if len(n.args) != 1 or not isinstance(n.args[0], ast.Constant) or n.args[0].value not in (0, 1):
                    raise ValueError('unexpected stream ordinal')
                n.args[0] = ast.Constant(value=0)
            return n
    import copy
    def stable(value):
        if isinstance(value, ast.AST):
            if getattr(value, 'type_params', []):
                raise ValueError('nonempty generic type parameters are not eligible')
            return [type(value).__name__, {k: stable(v) for k,v in ast.iter_fields(value) if k != 'type_params'}]
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    return digest(json.dumps(stable(Rank().visit(copy.deepcopy(node))), sort_keys=True, separators=(',', ':')).encode())

def runner_method(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Runner':
            return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'call')
    return None

def instrument_method(node, site):
    import copy
    node = copy.deepcopy(node)
    new = []
    count = 0
    for statement in node.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == INDEX:
            call = statement.value
            if len(call.args) != 2 or call.keywords:
                raise ValueError('Index signature differs')
            name = f'_ap544_operands_{count}'
            # Each original operand expression is evaluated ONCE, in original order.
            new.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Tuple(elts=call.args, ctx=ast.Load())))
            new.append(ast.Expr(value=ast.Call(func=ast.Name(id='_ap544_observe', ctx=ast.Load()), args=[ast.Constant(site), ast.Name(id=name, ctx=ast.Load())], keywords=[])))
            call.args = [ast.Starred(value=ast.Name(id=name, ctx=ast.Load()), ctx=ast.Load())]
            new.append(statement)
            new.append(ast.Delete(targets=[ast.Name(id=name, ctx=ast.Del())]))
            count += 1
        else:
            new.append(statement)
    if count != 1:
        raise ValueError(f'expected one original Index, found {count}')
    node.body = new
    node.name = '_ap544_runner_call'
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))

class Observer:
    def __init__(self, config, emit=print, clock=time.monotonic):
        self.config = config
        self.emit = emit
        self.clock = clock
        self.calls = self.captured = self.bytes = 0
        self.copy_seconds = 0.0
        self.exhausted = False
        self.lock = threading.Lock()
        self.sites = {}

    def output(self, record):
        raw = 'AP544 ' + json.dumps(record, separators=(',', ':'), sort_keys=True)
        size = len(raw.encode()) + 1
        if self.bytes + size > self.config['max_bytes']:
            self.exhausted = True
            return
        self.bytes += size
        self.emit(raw, flush=True)

    def __call__(self, site, operands):
        import torch
        with self.lock:
            self.calls += 1
            # Reserve32768 bytes for this bounded record plus the final stop marker.
            if self.exhausted:
                return
            if (self.captured >= self.config['max_calls'] or self.bytes + 32768 > self.config['max_bytes'] or self.copy_seconds >= self.config['max_copy_seconds']):
                self.exhausted = True
                self.output({'event':'CAP_EXHAUSTED','calls':self.calls,'captured':self.captured,'copy_seconds':self.copy_seconds,'bytes_before':self.bytes,'site_counts':self.sites})
                return
            tensor, indices = operands
            if tuple(tensor.shape) != (1024,151936) or tensor.dtype != torch.float32 or len(indices) != 2 or any(x.dtype != torch.int64 or tuple(x.shape) != (1024,) for x in indices):
                raise RuntimeError('AP544 guarded Index operand signature changed')
            seq = self.calls
            self.captured += 1
            self.sites[site] = self.sites.get(site,0) + 1
            def meta(x):
                return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'device':str(x.device),'ptr':x.data_ptr(),'storage_ptr':x.untyped_storage().data_ptr(),'storage_bytes':x.untyped_storage().nbytes(),'offset':x.storage_offset(),'version':x._version}
            md = {'tensor':meta(tensor),'indices':[meta(x) for x in indices]}
            self.output({'event':'ENTER_PRE_INDEX','site':site,'seq':seq,'pid':os.getpid(),'rank':os.environ.get('RANK'),'metadata':md})
            start = self.clock()
            # Synchronous D2H copies complete before the original Index is invoked.
            # They never replace the original operands or mutate their storage.
            values = [x.detach().to(device='cpu', copy=True).tolist() for x in indices]
            elapsed = self.clock() - start
            self.copy_seconds += elapsed
            blobs = [struct.pack('<1024q', *v) for v in values]
            record = {'event':'CAPTURE_PRE_INDEX','site':site,'seq':seq,'pid':os.getpid(),'rank':os.environ.get('RANK'),'metadata':md,'indices_le_i64_b64':[base64.b64encode(b).decode() for b in blobs],'indices_sha256':[digest(b) for b in blobs],'bad_counts':[sum(not(-1024<=x<1024) for x in values[0]),sum(not(-151936<=x<151936) for x in values[1])],'copy_seconds':elapsed,'cumulative_copy_seconds':self.copy_seconds}
            self.output(record)
            if self.bytes > self.config['max_bytes']:
                raise RuntimeError('AP544 encoded record exceeded reserved byte bound')

_INSTALLED = None

def patch_module(module, source, allowed, observer):
    if getattr(module, '_ap544_patched', False):
        return False
    method = runner_method(source)
    if method is None:
        return False
    fingerprint = canonical_method(method)
    if fingerprint not in allowed:
        return False
    if not hasattr(module, 'runner') or not hasattr(module, 'call'):
        raise RuntimeError('AP544 expected generated Runner exports')
    site = allowed[fingerprint]
    tree = instrument_method(method, site)
    module.__dict__['_ap544_observe'] = observer
    exec(compile(tree, module.__file__, 'exec'), module.__dict__)
    module.runner.call = types.MethodType(module.__dict__['_ap544_runner_call'], module.runner)
    module.call = module.runner.call
    module._ap544_patched = True
    module._ap544_original_source_sha256 = digest(source.encode())
    observer.output({'event':'MODULE_PATCHED','site':site,'method_sha256':fingerprint,'source_sha256':digest(source.encode()),'transform_sha256':digest(ast.dump(tree).encode())})
    return True

def install(config_path=None):
    global _INSTALLED
    if _INSTALLED is not None:
        return _INSTALLED
    import torch._inductor.codecache as cache
    config_path = config_path or os.environ.get('MILES_INDEX_OBSERVER544_CONFIG')
    if not config_path:
        raise RuntimeError('AP544 diagnostic requires explicit guarded configuration')
    raw = Path(config_path).read_bytes()
    if len(raw) > 16384:
        raise RuntimeError('AP544 config too large')
    config = json.loads(raw)
    if set(config) != {'runtime_hashes','allowed_methods','max_calls','max_bytes','max_copy_seconds'}:
        raise RuntimeError('AP544 unexpected config schema')
    if type(config['max_calls']) is not int or not 1 <= config['max_calls'] <= 4096:
        raise RuntimeError('AP544 call bound')
    if type(config['max_bytes']) is not int or not 65536 <= config['max_bytes'] <= 134217728:
        raise RuntimeError('AP544 byte bound')
    if type(config['max_copy_seconds']) not in (int,float) or not 0 < config['max_copy_seconds'] <= 300:
        raise RuntimeError('AP544 copy-time admission bound')
    root = Path(cache.__file__).resolve().parent.parent
    for rel, expected in config['runtime_hashes'].items():
        if rel not in ('_inductor/codecache.py','_inductor/compile_fx.py','_inductor/output_code.py','version.py') or digest((root/rel).read_bytes()) != expected:
            raise RuntimeError('AP544 runtime source hash mismatch: '+rel)
    if set(config['runtime_hashes']) != {'_inductor/codecache.py','_inductor/compile_fx.py','_inductor/output_code.py','version.py'}:
        raise RuntimeError('AP544 incomplete runtime closure')
    allowed = config['allowed_methods']
    if (type(allowed) is not dict or len(allowed) != 4 or
        any(type(k) is not str or len(k) != 64 or any(c not in '0123456789abcdef' for c in k) for k in allowed) or
        sorted(allowed.values()) != ['backward','backward','producer','producer']):
        raise RuntimeError('AP552 requires both exact observed ordinal variants of both original wrapper methods')
    # Old bound callables cannot be repaired by replacing module attributes.
    # Refuse preexisting matching modules before mutating the runtime hook.
    existing = list(cache.PyCodeCache.modules) + list(cache.PyCodeCache.modules_no_attr.values())
    for module in existing:
        path = getattr(module, '__file__', None)
        if path and Path(path).exists() and Path(path).stat().st_size <= 1048576:
            method = runner_method(Path(path).read_text())
            if method is not None and canonical_method(method) in allowed:
                raise RuntimeError('AP544 must install before any matching generated module loads')
    observer = Observer(config)
    original = cache.PyCodeCache.load_by_key_path.__func__
    def load(cls, key, path, linemap=None, attrs=None):
        module = original(cls, key, path, linemap, attrs)
        if not getattr(module, '_ap544_patched', False):
            source_path = Path(path)
            if source_path.stat().st_size <= 1048576:
                patch_module(module, source_path.read_text(), allowed, observer)
        return module
    cache.PyCodeCache.load_by_key_path = classmethod(load)
    _INSTALLED = observer
    return observer
