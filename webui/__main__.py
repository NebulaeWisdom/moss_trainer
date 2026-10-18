"""Start the local web workbench from the repository's parent workspace."""
from __future__ import annotations

import argparse
import ipaddress


def main() -> None:
    parser = argparse.ArgumentParser(description="MOSS local training web workbench (loopback only).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host == "localhost":
        args.host = "127.0.0.1"
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            raise ValueError
    except ValueError:
        parser.error("--host must be a loopback IP address or localhost; remote binding is not supported.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    import uvicorn
    uvicorn.run("moss_trainer.webui.app:create_app", factory=True, host=args.host, port=args.port, proxy_headers=False, server_header=False)


if __name__ == "__main__":
    main()
