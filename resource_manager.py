from __future__ import annotations
import platformdirs
import io
import pathlib as pl
import urllib.request
import os

from tqdm import tqdm


class ResourceManager:
    root: pl.Path

    def __init__(self, root: pl.Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, filename: str, url: str) -> pl.Path:
        path = self.path(filename)
        if not path.exists():
            self.download(filename, url)

        return path

    def download(self, filename: str, url: str) -> pl.Path:
        filepath = self.path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(url) as http:
            length = http.getheader("content-length")
            blocksize = None
            t = None
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)
                t = tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    desc=filepath.name,
                    total=length,
                )

            buf = io.BytesIO()
            size = 0
            with open(filepath, "wb") as fp:
                while True:
                    block = http.read(blocksize)
                    if not block:
                        break
                    buf.write(block)
                    size += len(block)
                    if t:
                        t.update(len(block))

                    # Write to file every 10MB
                    if len(buf.getbuffer()) >= 10000000:
                        fp.write(buf.getbuffer())
                        buf.seek(0)
                        buf.truncate()

                # After all the data is downloaded, make sure to flush the remaining buffer content to file
                if len(buf.getbuffer()) > 0 :
                    fp.write(buf.getbuffer())

        return filepath

    def path(self, filename: str) -> pl.Path:
        return self.root.joinpath(filename)


RESOURCE_MANAGER = ResourceManager(pl.Path(os.path.abspath(__file__)).parent / "models")

def get_datadir() -> pl.Path:
    return platformdirs.user_data_path("imagetranslator", "aleb2000")
