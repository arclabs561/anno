import http.client
import io
import json
import os
from email.message import Message
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib import error, request, response

import oracle_gateway as gateway


def envelope(answer='{"judgment": ["café", 7]}'):
    return {
        "model": "example/resolved",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 23, "completion_tokens": 7, "cost": 0.0002},
    }


class FixtureHTTPS(request.HTTPSHandler):
    """Supply a local response while exercising urllib's real handler chain."""

    def __init__(self):
        super().__init__()
        self.requests = []
        self.raw = json.dumps(envelope()).encode()
        self.code = 200
        self.headers = Message()

    def https_open(self, req):
        self.requests.append(req)
        if len(self.requests) > 1:
            raise AssertionError("unexpected second request")
        stream = io.BytesIO(self.raw)
        stream.read = self.read = Mock(wraps=stream.read)
        reply = response.addinfourl(stream, self.headers, req.full_url, self.code)
        reply.msg = "fixture response"
        return reply


class OracleGateway(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENROUTER_API_KEY": "fixture-secret"}, clear=True))
        self.https = FixtureHTTPS()
        build_opener = request.build_opener
        self.enterContext(patch.object(request, "build_opener",
                                       side_effect=lambda *handlers: build_opener(*handlers, self.https)))

    def call(self, prompt="Judge this café. Return JSON.", **overrides):
        params = {"model": "example/requested", "max_tokens": 120, "timeout": 7.5}
        return gateway.complete(prompt, **{**params, **overrides})

    def test_request_destination_content_and_reported_usage(self):
        os.environ["OPENROUTER_BASE_URL"] = "https://other.invalid"
        os.environ["UNRELATED"] = "fixture-private"
        text, model, inputs, outputs, cost = self.call()
        self.assertEqual((model, inputs, outputs, cost), ("example/resolved", 23, 7, 0.0002))
        self.assertEqual(text, envelope()["choices"][0]["message"]["content"])
        req, = self.https.requests
        self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual((req.get_method(), req.timeout), ("POST", 7.5))
        self.assertEqual(req.get_header("Authorization"), "Bearer fixture-secret")
        self.assertEqual(json.loads(req.data), {
            "model": "example/requested", "messages": [{"role": "user", "content": "Judge this café. Return JSON."}],
            "max_tokens": 120, "response_format": {"type": "json_object"}, "provider": {"require_parameters": True},
        })

    def test_optional_metadata_zero_and_text_preservation(self):
        answer = ' { "arbitrary": [null, {}, true] }\n'
        for metadata, expected in [
            ({}, (None, None, None, None)),
            ({"usage": None}, (None, None, None, None)),
            ({"usage": {"completion_tokens": 0, "cost": 0}}, (None, None, 0, 0.0)),
        ]:
            with self.subTest(metadata=metadata):
                self.https.requests.clear()
                self.https.raw = json.dumps({"choices": envelope(answer)["choices"], **metadata}).encode()
                self.assertEqual(self.call(), (answer, *expected))

    def test_native_array_omits_object_mode_and_preserves_array(self):
        answer = ' [ { "label": "yes" }, null ] \n'
        self.https.raw = json.dumps(envelope(answer)).encode()
        self.assertEqual(self.call(shape="array")[0], answer)
        req, = self.https.requests
        self.assertNotIn("response_format", json.loads(req.data))

    def test_native_root_must_match_in_both_modes(self):
        for shape, answer in (("array", "{}"), ("object", "[]"), ("array", "false")):
            with self.subTest(shape=shape, answer=answer):
                self.https.requests.clear()
                self.https.raw = json.dumps(envelope(answer)).encode()
                with self.assertRaisesRegex(gateway.OracleGatewayError, f"JSON {shape} answer"):
                    self.call(shape=shape)

    def test_key_file_has_precedence_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "fixture-key-file"
            os.environ["OPENROUTER_API_KEY_FILE"] = str(key)
            with self.assertRaisesRegex(gateway.OracleGatewayError, "^cannot read the OpenRouter key file$"):
                self.call()
            for contents in (b"", b"fixture\nheader", b"x" * 4097, b"\xff"):
                with self.subTest(contents_length=len(contents)):
                    key.write_bytes(contents)
                    with self.assertRaises(gateway.OracleGatewayError):
                        self.call()
            self.assertEqual(self.https.requests, [])
            key.write_bytes(b"from-fixture-file\n")
            self.call()
            self.assertEqual(self.https.requests[0].get_header("Authorization"), "Bearer from-fixture-file")

    def test_absent_credentials_and_invalid_request_parameters_do_not_dispatch(self):
        for params in ({"model": "x\nprivate"}, {"max_tokens": True}, {"timeout": float("inf")},
                       {"timeout": -1}, {"prompt": ""}, {"prompt": "\ud800"}, {"shape": "string"}):
            with self.subTest(params=params), self.assertRaises(gateway.OracleGatewayError):
                self.call(**params)
        with patch.object(gateway, "_MAX_REQUEST_BYTES", 32), self.assertRaises(gateway.OracleGatewayError):
            self.call(prompt="é" * 32)
        del os.environ["OPENROUTER_API_KEY"]
        with self.assertRaises(gateway.OracleGatewayError):
            self.call()
        self.assertEqual(self.https.requests, [])

    def test_redirects_are_refused_without_sending_another_request(self):
        self.https.headers["Location"] = "https://elsewhere.invalid/fixture"
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                self.https.code = code
                self.https.requests.clear()
                with self.assertRaisesRegex(gateway.OracleGatewayError, f"^OpenRouter returned HTTP {code}$"):
                    self.call()
                self.assertEqual(len(self.https.requests), 1)

    def test_http_and_transport_diagnostics_hide_private_content(self):
        self.https.raw = b"fixture-secret: private provider diagnostic"
        for status in (429, 202):
            self.https.requests.clear()
            self.https.code = status
            with self.assertRaisesRegex(gateway.OracleGatewayError, f"^OpenRouter returned HTTP {status}$"):
                self.call()
            self.https.read.assert_not_called()
        for exc in (error.URLError("fixture-secret"), http.client.IncompleteRead(b"fixture-secret"),
                    ValueError("fixture-private")):
            with patch.object(self.https, "https_open", side_effect=exc), self.assertRaisesRegex(
                gateway.OracleGatewayError, "^OpenRouter request failed$"
            ):
                self.call()

    def test_response_size_is_bounded(self):
        with patch.object(gateway, "_MAX_RESPONSE_BYTES", 30), self.assertRaisesRegex(
            gateway.OracleGatewayError, "response exceeds"
        ):
            self.call()
        self.https.read.assert_called_once_with(31)

    def test_incomplete_envelopes_and_non_object_answers_fail(self):
        choice = envelope()["choices"][0]
        bad_choices = [None, {}, [], [choice, choice], [None]]
        for field in ({"finish_reason": "length"}, {"finish_reason": None}, {"error": {"private": True}},
                      {"message": None}, {"message": {"role": "user", "content": "{}"}},
                      {"message": {**choice["message"], "refusal": "fixture-private"}}):
            bad_choices.append([{**choice, **field}])
        bad_bodies = [{"choices": choices} for choices in bad_choices]
        bad_bodies.extend(envelope(answer) for answer in ([], "", "[]", "null", '{"x":NaN}', "```json\n{}\n```"))
        bad_bodies.extend([[], {"error": "fixture-private", **envelope()}, {**envelope(), "model": "bad\nmodel"}])
        for raw in [b"\xff", b"<html>fixture-private</html>", *(json.dumps(body).encode() for body in bad_bodies)]:
            with self.subTest(raw=raw):
                self.https.requests.clear()
                self.https.raw = raw
                with self.assertRaises(gateway.OracleGatewayError) as raised:
                    self.call()
                self.assertNotIn("fixture-private", str(raised.exception))

    def test_invalid_usage_cannot_be_reported(self):
        for usage in ([], {"prompt_tokens": -1}, {"completion_tokens": True}, {"prompt_tokens": 1.2},
                      {"completion_tokens": 2**63}, {"cost": -1}, {"cost": True}, {"cost": 10**400},
                      {"cost": "0.01"}, {"cost": float("inf")}):
            with self.subTest(usage=usage):
                self.https.requests.clear()
                self.https.raw = json.dumps({**envelope(), "usage": usage}).encode()
                with self.assertRaises(gateway.OracleGatewayError):
                    self.call()


if __name__ == "__main__":
    unittest.main()
