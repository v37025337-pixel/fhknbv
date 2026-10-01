"""Explicit HTTP GET capability; remote documents enter cognition as data.

Uses the inherited proxy and verified TLS. No cookies, credentials from URLs,
POST requests, remote tools, or execution of fetched code are introduced.
"""
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import ipaddress
import json
import math
from pathlib import Path
import socket
import time
import zlib
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .cognition import atomic_json


class WebError(RuntimeError):
    pass


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title = [], []
        self.suppressed = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'noscript', 'template'}:
            self.suppressed += 1
        if tag == 'title':
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'noscript', 'template'}:
            self.suppressed = max(0, self.suppressed - 1)
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)
            if self.in_title:
                self.title.append(data)


def _scalar(value):
    return (value is None or type(value) in (str, bool, int)
            or type(value) is float and math.isfinite(value))


def _reject_constant(value):
    raise ValueError(f'nonfinite JSON value: {value}')


def _json_fields(document):
    fields = []

    def walk(value, path, depth):
        if len(fields) >= 48 or depth > 4:
            return
        if _scalar(value):
            if type(value) is str:
                value = value[:256]
            fields.append([path[:256], value])
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f'{path}.{key}' if path else key, depth + 1)
                if len(fields) >= 48:
                    break
        elif isinstance(value, list):
            for index, item in enumerate(value[:8]):
                walk(item, f'{path}[{index}]', depth + 1)
    walk(document, '', 0)
    return fields


class _Redirect(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def __init__(self, validate):
        self.validate = validate

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.validate(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebReader:
    def __init__(self, *, timeout=10., max_bytes=262144, max_chars=12000,
                 opener=None, resolver=socket.getaddrinfo):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError('timeout must be a finite number in (0, 30]')
        if type(max_bytes) is not int or not 0 < max_bytes <= 524288:
            raise ValueError('max_bytes must be an integer in 1..524288')
        if type(max_chars) is not int or not 0 < max_chars <= 12000:
            raise ValueError('max_chars must be an integer in 1..12000')
        self.timeout, self.max_bytes, self.max_chars = timeout, max_bytes, max_chars
        self.resolver = resolver
        self.redirect_handler = _Redirect(self.validate_url)
        # build_opener preserves environment proxy routing and TLS verification.
        self.opener = opener if opener is not None else build_opener(self.redirect_handler)

    def validate_url(self, url):
        if not isinstance(url, str) or not url or len(url) > 4096 or any(c.isspace() or ord(c) < 32 for c in url):
            raise ValueError('URL must be a nonempty HTTP(S) URL of at most 4096 characters')
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('only HTTP(S) URLs are supported')
        if parsed.username is not None or parsed.password is not None:
            raise ValueError('credentials in URLs are not supported')
        host = parsed.hostname.rstrip('.').lower()
        if host == 'localhost' or host.endswith(('.localhost', '.local')):
            raise ValueError('destination must be a public internet host')
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80)
        if port == 0:
            raise ValueError('destination port must be in 1..65535')
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            addresses = [ipaddress.ip_address(item[4][0])
                         for item in self.resolver(host, port, type=socket.SOCK_STREAM)]
        if not addresses or any(not ip.is_global or ip.is_multicast for ip in addresses):
            raise ValueError('destination must resolve to public internet addresses')
        return url

    def fetch(self, url):
        self.validate_url(url)
        started = time.monotonic()
        request = Request(url, headers={'User-Agent': 'DIGITAL-MIND/0.3.1 (public-data-reader)',
                          'Accept': 'text/html, application/json, text/plain, application/xml;q=0.8',
                          'Accept-Encoding': 'identity'}, method='GET')
        with self.opener.open(request, timeout=self.timeout) as response:
            final_url = self.validate_url(response.geturl())
            status = response.status
            if not 200 <= status < 300:
                raise WebError(f'HTTP status {status}')
            compression = response.headers.get('Content-Encoding', 'identity').strip().lower()
            if compression not in {'', 'identity', 'gzip', 'deflate'}:
                raise WebError(f'unsupported content encoding: {compression}')
            mime = response.headers.get_content_type()
            if not (mime.startswith('text/') or mime == 'application/json'
                    or mime.endswith('+json') or mime in {'application/xml', 'application/xhtml+xml'}):
                raise WebError(f'unsupported response media type: {mime}')
            wire = response.read(self.max_bytes + 1)
            if len(wire) > self.max_bytes:
                raise WebError('response exceeds the byte limit')
            encoding = response.headers.get_content_charset() or 'utf-8'
        raw = wire
        if compression in {'gzip', 'deflate'}:
            decoder = zlib.decompressobj(zlib.MAX_WBITS + (16 if compression == 'gzip' else 0))
            try:
                raw = decoder.decompress(wire, self.max_bytes + 1)
            except zlib.error as exc:
                raise WebError('invalid compressed response') from exc
            if len(raw) > self.max_bytes or decoder.unconsumed_tail:
                raise WebError('decoded response exceeds the byte limit')
            if not decoder.eof or decoder.unused_data:
                raise WebError('incomplete or concatenated compressed response')
        try:
            decoded = raw.decode(encoding, errors='replace')
        except LookupError as exc:
            raise WebError('unsupported response charset') from exc
        title, fields = '', []
        if mime in {'text/html', 'application/xhtml+xml'}:
            parser = _Text()
            parser.feed(decoded)
            parser.close()
            text = ' '.join(' '.join(parser.parts).split())
            title = ' '.join(' '.join(parser.title).split())[:512]
        else:
            text = decoded
        if mime == 'application/json' or mime.endswith('+json'):
            try:
                document = json.loads(decoded, parse_constant=_reject_constant)
            except (ValueError, RecursionError) as exc:
                raise WebError('invalid JSON response') from exc
            fields = _json_fields(document)
        timestamp = datetime.now(timezone.utc).isoformat()
        checksum = hashlib.sha256(raw).hexdigest()
        identifier = hashlib.sha256((final_url + timestamp + checksum).encode()).hexdigest()[:24]
        return {'id': identifier, 'requested_url': url, 'url': final_url,
                'retrieved_at': timestamp, 'http_status': status, 'media_type': mime,
                'title': title, 'text': text[:self.max_chars], 'json_fields': fields,
                'truncated': len(text) > self.max_chars, 'bytes_received': len(wire), 'decoded_bytes': len(raw),
                'content_sha256': checksum, 'elapsed_seconds': time.monotonic() - started,
                'trust': 'unverified_external_data'}


class WebConnection:
    max_sources = 16

    def __init__(self, mind, reader, *, memory_path=None):
        self.mind, self.reader = mind, reader
        self.memory_path = Path(memory_path) if memory_path is not None else None
        self.records = deque(maxlen=self.max_sources)
        self.successful_fetches = 0
        self.failed_fetches = 0
        self.last_error = None
        if self.memory_path is not None and self.memory_path.exists():
            if self.memory_path.stat().st_size > 2_000_000:
                raise ValueError('web memory exceeds the byte limit')
            data = json.loads(self.memory_path.read_text(encoding='utf-8'), parse_constant=_reject_constant)
            if not isinstance(data, dict) or data.get('format') != 'digital-mind-web-memory-v1':
                raise ValueError('unsupported web memory format')
            records = data.get('sources')
            if not isinstance(records, list) or len(records) > self.max_sources:
                raise ValueError('invalid web memory sources')
            for record in records:
                self._check_record(record)
                self.records.append(copy.deepcopy(record))

    @staticmethod
    def _check_record(record):
        strings = {'id': 24, 'requested_url': 4096, 'url': 4096, 'retrieved_at': 64,
                   'media_type': 128, 'title': 512, 'text': 12000, 'content_sha256': 64}
        if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or len(record[k]) > n
                                               for k, n in strings.items()):
            raise ValueError('invalid web memory record')
        if set(record) - (set(strings) | {'trust', 'json_fields', 'http_status', 'truncated',
                                       'bytes_received', 'decoded_bytes', 'elapsed_seconds'}):
            raise ValueError('unsupported web memory fields')
        if record.get('trust') != 'unverified_external_data':
            raise ValueError('web memory must remain unverified external data')
        for key, width in (('id', 24), ('content_sha256', 64)):
            if len(record[key]) != width or any(c not in '0123456789abcdef' for c in record[key]):
                raise ValueError('invalid web memory identifier or hash')
        fields = record.get('json_fields')
        if (not isinstance(fields, list) or len(fields) > 48
                or any(not isinstance(f, list) or len(f) != 2 or not isinstance(f[0], str)
                       or len(f[0]) > 256 or not _scalar(f[1])
                       or isinstance(f[1], str) and len(f[1]) > 256 for f in fields)):
            raise ValueError('invalid web memory fields')
        if type(record.get('http_status')) is not int or not 200 <= record['http_status'] < 300:
            raise ValueError('invalid web memory HTTP status')
        if type(record.get('truncated')) is not bool:
            raise ValueError('invalid web memory truncation flag')
        if type(record.get('bytes_received')) is not int or not 0 <= record['bytes_received'] <= 524288:
            raise ValueError('invalid web memory byte count')
        if 'decoded_bytes' in record and (type(record['decoded_bytes']) is not int
                                         or not 0 <= record['decoded_bytes'] <= 524288):
            raise ValueError('invalid web memory decoded byte count')
        elapsed = record.get('elapsed_seconds')
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid web memory duration')

    @staticmethod
    def augment_task(task, record):
        task = copy.deepcopy(task)
        observation = {'source': 'internet', **copy.deepcopy(record)}
        facts = [['web_fetched', record['id']], ['web_url', record['id'], record['url']],
                 ['web_source_unverified', record['id']]]
        facts.extend(['web_field', record['id'], key, value] for key, value in record['json_fields'])
        task['observations'] = list(task.get('observations', [])) + [observation]
        task['facts'] = list(task.get('facts', [])) + facts
        task.setdefault('context', 'internet')
        return task

    def fetch(self, url):
        try:
            record = self.reader.fetch(url)
            self._check_record(record)
        except Exception as exc:
            self.failed_fetches += 1
            self.last_error = {'type': type(exc).__name__, 'message': str(exc)[:512]}
            raise
        records = deque(self.records, maxlen=self.max_sources)
        records.append(record)
        if self.memory_path is not None:
            atomic_json(self.memory_path, {'format': 'digital-mind-web-memory-v1', 'sources': list(records)})
        self.records = records
        result = self.mind.process(self.augment_task(
            {'queries': [['web_fetched', record['id']]], 'candidate_actions': ['observe_source']}, record))
        self.successful_fetches += 1
        self.last_error = None
        return {'source': copy.deepcopy(record), 'kernel': {key: result.get(key)
                for key in ('cycle', 'action', 'reason', 'inferences')},
                'memory_persisted': self.memory_path is not None}

    def lookup(self, source_id):
        if not isinstance(source_id, str):
            raise ValueError('source_id must be a string')
        for record in self.records:
            if record['id'] == source_id:
                return copy.deepcopy(record)
        raise KeyError('web source is not in the retained memory')

    def memory(self):
        return {'sources': [{k: v for k, v in record.items() if k not in {'text', 'json_fields'}}
                            for record in copy.deepcopy(list(self.records))]}

    def status(self):
        return {'attached': True, 'transport': 'HTTP(S) GET', 'successful_fetches': self.successful_fetches,
                'failed_fetches': self.failed_fetches, 'cached_sources': len(self.records),
                'max_sources': self.max_sources, 'last_error': copy.deepcopy(self.last_error),
                'memory_persistence': self.memory_path is not None,
                'max_response_bytes': self.reader.max_bytes, 'max_text_chars': self.reader.max_chars,
                'socket_timeout_seconds': self.reader.timeout}
