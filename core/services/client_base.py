from typing import Dict, Optional

import aiohttp
from aiohttp import ClientResponse


class ClientBase:
    base_url = None

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def post(self, endpoint: str, payload: Optional[Dict] = None, params: Optional[Dict] = None,
                   auth: Optional[aiohttp.BasicAuth] = None):
        """
        Post request to the backend API.
        :param endpoint:
        :param payload:
        :param params:
        :param auth:
        :return:
        """
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        response: ClientResponse = await self.session.post(url, json=payload, params=params, headers=headers, auth=auth)
        return await self._process_response(response)

    async def get(self, endpoint: str, params: Optional[Dict] = None, auth: Optional[aiohttp.BasicAuth] = None):
        """
        Get request to the backend API.
        :param endpoint:
        :param params:
        :param auth:
        :return:
        """
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint}"
        response = await self.session.get(url, params=params, headers={"accept": "application/json"}, auth=auth)
        return await self._process_response(response)

    @staticmethod
    async def _process_response(response):
        if response.status >= 400:
            text = await response.text()
            print(f"Error: {response.status} - {text}")
            return {"error": text, "status": response.status}
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            return await response.json()
        else:
            text = await response.text()
            print(f"Warning: Unexpected content type: {content_type}")
            print(f"Response text: {text}")
            return {"content": text, "content_type": content_type}

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def __del__(self):
        if self.session and not self.session.closed:
            import asyncio
            asyncio.create_task(self.close())
