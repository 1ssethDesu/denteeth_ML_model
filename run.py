#!/usr/bin/env python3
"""
Startup script for the Dental X-Ray Detection API
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Dental X-Ray Detection API")
    parser.add_argument(
        "--host", 
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        default=os.getenv("DEBUG", "False").lower() == "true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO").lower(),
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Add the project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    print(f"Starting Dental X-Ray Detection API...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print(f"Workers: {args.workers}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,
        access_log=True
    )

if __name__ == "__main__":
    main() 