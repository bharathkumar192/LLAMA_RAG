import logging
import os
import shutil
import subprocess
import argparse

import torch
from flask import Flask, jsonify, request, stream_with_context, Response
from main import load_model
import asyncio
from constants import *
from utils import *
from pydantic import BaseModel
from typing import List, Any
from flask_ngrok import run_with_ngrok
import threading
import getpass
from pyngrok import ngrok, conf



result =  print(chat("what is Null Value", []))