"""Read AP544 JSON; recover only one adjacent same-Ray-prefix string split.

This is log framing, not proof of complete capture or numerical correctness.
"""
import json
import re

_RAY = re.compile(r'^(?:\x1b\[[0-9;]*m)?\([^()\r\n]+ pid=(?P<pid>[0-9]+)\)(?:\x1b\[[0-9;]*m)? ')
_MAX_SPLIT_BYTES = 32768


def capture_records(lines):
    lines = iter(lines)
    for line in lines:
        start = line.find('AP544 ')
        if start < 0:
            continue
        body = line[start + len('AP544 '):].rstrip('\r\n')
        try:
            record = json.loads(body)
        except json.JSONDecodeError as error:
            # Actual940 splits a long base64 string into two Ray-prefixed lines.
            # Other malformed JSON, unprefixed fragments and interleaving fail closed.
            prefix = _RAY.match(line)
            if (error.msg != 'Unterminated string starting at' or prefix is None
                    or prefix.end() != start or len(body.encode()) > _MAX_SPLIT_BYTES):
                raise
            following = next(lines, None)
            if following is None or not following.startswith(prefix.group(0)):
                raise ValueError('AP544 continuation absent or different Ray identity') from error
            continuation = following[len(prefix.group(0)):].rstrip('\r\n')
            if 'AP544 ' in continuation or len((body + continuation).encode()) > _MAX_SPLIT_BYTES:
                raise ValueError('AP544 continuation ambiguous or over byte bound') from error
            record = json.loads(body + continuation)
            if (type(record.get('pid')) is not int
                    or record['pid'] != int(prefix.group('pid'))
                    or record.get('event') != 'CAPTURE_PRE_INDEX'):
                raise ValueError('AP544 split capture identity mismatch')
        yield record
