import os
import unittest

from fastapi.testclient import TestClient

from nsys_llm_explainer.api import app


class ApiAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_key = os.environ.get("NSYS_API_KEY")
        self.client = TestClient(app)
        self.report_json = b'{"tool":{"version":"0.3.1"},"metrics":{},"findings":[],"warnings":[]}'

    def tearDown(self) -> None:
        if self._old_key is None:
            os.environ.pop("NSYS_API_KEY", None)
        else:
            os.environ["NSYS_API_KEY"] = self._old_key

    def _post_json(self, headers=None):
        return self.client.post(
            "/v1/analyze/json",
            files={"file": ("report.json", self.report_json, "application/json")},
            headers=headers or {},
        )

    def test_public_mode_allows_request_without_api_key(self) -> None:
        os.environ.pop("NSYS_API_KEY", None)
        response = self._post_json()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("summary", payload)
        self.assertIn("report", payload)

    def test_api_key_mode_rejects_missing_key(self) -> None:
        os.environ["NSYS_API_KEY"] = "supersecret"
        response = self._post_json()
        self.assertEqual(response.status_code, 401)

    def test_api_key_mode_accepts_x_api_key(self) -> None:
        os.environ["NSYS_API_KEY"] = "supersecret"
        response = self._post_json(headers={"x-api-key": "supersecret"})
        self.assertEqual(response.status_code, 200)

    def test_api_key_mode_accepts_bearer_token(self) -> None:
        os.environ["NSYS_API_KEY"] = "supersecret"
        response = self._post_json(headers={"Authorization": "Bearer supersecret"})
        self.assertEqual(response.status_code, 200)

    def test_healthz_is_public_even_when_api_key_enabled(self) -> None:
        os.environ["NSYS_API_KEY"] = "supersecret"
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("auth_enabled"))


if __name__ == "__main__":
    unittest.main()
