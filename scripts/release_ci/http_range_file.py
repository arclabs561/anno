"""Seekable HTTP reads over validated byte ranges, for large publisher files."""

import io
from urllib.request import Request


class RemoteFile(io.RawIOBase):
    def __init__(self, url, size, fetch):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0
        self.fetch = fetch

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = {0: 0, 1: self.position, 2: self.size}[whence] + offset
        if self.position < 0:
            raise OSError("Seek before start of pinned file")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        start = self.position
        stop = self.size if size < 0 else min(self.size, start + size)
        if start >= stop:
            return b""
        # Include the exact range in the query as well. Shared HTTP caches can
        # otherwise return an old response to a different Range request.
        request = Request(
            f"{self.url}?range={start}-{stop - 1}",
            headers={
                "Range": f"bytes={start}-{stop - 1}",
                "Accept-Encoding": "identity",
            },
        )

        def checked(response):
            if (
                response.status != 206
                or response.headers.get("Content-Range")
                != f"bytes {start}-{stop - 1}/{self.size}"
            ):
                raise OSError("Publisher did not return the requested byte range")
            data = response.read(stop - start + 1)
            if len(data) != stop - start:
                raise OSError("Truncated byte range")
            return data

        data = self.fetch(request, checked)
        self.position = stop
        return data
