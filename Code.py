#!/usr/bin/env python3
"""
Ollama Coder v2 - a local AI coding agent for Ollama.

* Streaming chat with visible thinking, rendered markdown and tool cards
* Tools: files, search, edit, shell, Python, C/C++ (gcc/g++), date/time
* Research tools: web_research / web_search / fetch_url (approved sites or open web)
* Modern dark tkinter UI - standard library only, no pip packages needed
"""

import ast
import concurrent.futures
import difflib
import fnmatch
import html
import inspect
import ipaddress
import json
import math
import os
import platform
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import OrderedDict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

APP = "Ollama Coder"
VERSION = "2.0"
CONFIG_PATH = Path.home() / ".ollama_coder.json"

DEFAULT_LINKS = [
    "https://docs.python.org/3/",
    "https://en.cppreference.com/",
    "https://www.lua.org/manual/5.4/",
    "https://gcc.gnu.org/onlinedocs/",
    "https://developer.mozilla.org/",
    "https://learn.microsoft.com/",
    "https://stackoverflow.com/",
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://pypi.org/",
]

DEFAULTS = {
    "host": "http://localhost:11434",
    "model": "glm-4.7-flash",
    "workspace": str(Path.cwd() / "workspace"),
    "temperature": 0.4,
    "num_ctx": 16384,
    "keep_alive": "30m",
    "timeout": 60,
    "max_steps": 30,
    "think": True,
    "show_thinking": True,
    "auto_approve": False,
    "confirm_writes": False,
    "web_mode": "approved",          # "approved" | "open" | "off"
    "links": DEFAULT_LINKS,
    "custom_instructions": "",
}

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".idea", ".vscode"}


def load_config():
    cfg = dict(DEFAULTS)
    cfg["links"] = list(DEFAULT_LINKS)
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if saved.get("provider") in ("openai", "openrouter"):
            saved.pop("model", None)         # old cloud model name - fall back to the Ollama default
        for key in DEFAULTS:
            if key in saved:
                cfg[key] = saved[key]
    except Exception:
        pass
    if not isinstance(cfg["links"], list):
        cfg["links"] = list(DEFAULT_LINKS)
    if cfg["web_mode"] not in ("approved", "open", "off"):
        cfg["web_mode"] = "approved"
    return cfg


def save_config(cfg):
    """Only known keys are written, so old API keys disappear from the file."""
    try:
        data = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print("Could not save config:", e)


def now_info():
    n = datetime.now().astimezone()
    z = n.strftime("%z") or "+0000"
    offset = "UTC%s%s:%s" % (z[0], z[1:3], z[3:5])
    return {
        "pretty": n.strftime("%A, %d %B %Y, %H:%M:%S"),
        "date": n.strftime("%Y-%m-%d"),
        "time": n.strftime("%H:%M:%S"),
        "year": n.year,
        "weekday": n.strftime("%A"),
        "tz": "%s, %s" % (n.tzname() or "local time", offset),
        "iso": n.isoformat(timespec="seconds"),
        "doy": n.timetuple().tm_yday,
        "week": n.isocalendar()[1],
        "epoch": int(n.timestamp()),
    }


def clip(s, n=8000):
    if len(s) <= n:
        return s
    head = n * 2 // 3
    tail = n - head
    return s[:head] + "\n... [%d chars truncated] ...\n" % (len(s) - n) + s[-tail:]


def _bool(v, default=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except Exception:
            pass
    return s.split() if s else []


# ----------------------------------------------------------------------------
# Tool schemas (descriptions are written for the model - they matter!)
# ----------------------------------------------------------------------------
def _t(name, desc, props, req=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(req)}}}


def _S(d):
    return {"type": "string", "description": d}


def _I(d):
    return {"type": "integer", "description": d}


def _B(d):
    return {"type": "boolean", "description": d}


def _L(d):
    return {"type": "array", "items": {"type": "string"}, "description": d}


TOOLS = [
    _t("get_datetime",
       "Get the exact current local date, time, weekday and timezone. Call this whenever a task depends on 'now' "
       "or you want to re-check the time during a long session.", {}),
    _t("list_files",
       "List files and folders in the workspace (with sizes). Use this first to see what exists. "
       "path is relative to the workspace root ('.' = root). recursive=true shows the whole tree (max 300 entries).",
       {"path": _S("Folder relative to the workspace, default '.'"),
        "recursive": _B("List subfolders too")}),
    _t("read_file",
       "Read a text file with line numbers (max 400 lines per call). Use start_line/end_line to page through "
       "large files. ALWAYS read a file before editing it.",
       {"path": _S("File path relative to the workspace"),
        "start_line": _I("First line to read (1-based), default 1"),
        "end_line": _I("Last line to read")}, ["path"]),
    _t("search_files",
       "Search for text inside workspace files (like grep). Returns 'path:line: text'. Use it to locate "
       "functions, variables, TODOs or error strings before reading files.",
       {"pattern": _S("Text to find (or a regex if regex=true)"),
        "path": _S("Folder or file to search, default '.'"),
        "glob": _S("Only files matching this name pattern, e.g. '*.py' or '*.c'"),
        "regex": _B("Treat pattern as a regular expression"),
        "ignore_case": _B("Case-insensitive search (default true)")}, ["pattern"]),
    _t("analyse_file",
       "Summarise a file: size, language, line count, functions/classes/imports, TODOs, and a syntax check for "
       "Python. Good for getting an overview before reading.",
       {"path": _S("File path relative to the workspace")}, ["path"]),
    _t("write_file",
       "Create a new file or completely overwrite an existing one. 'content' must be the COMPLETE file - never "
       "use placeholders such as '... rest unchanged ...'. Parent folders are created automatically. For small "
       "changes to an existing file use edit_file instead. Python files are syntax-checked automatically.",
       {"path": _S("File path relative to the workspace"),
        "content": _S("The full file content")}, ["path", "content"]),
    _t("edit_file",
       "Targeted edit: replace the text 'old' with 'new' in an existing file. 'old' must match the file EXACTLY "
       "(copy it from read_file without the line-number prefix) and must be unique - include 2-3 surrounding "
       "lines if needed, or set replace_all=true to change every occurrence.",
       {"path": _S("File path relative to the workspace"),
        "old": _S("Exact existing text to replace"),
        "new": _S("Replacement text"),
        "replace_all": _B("Replace every occurrence instead of requiring a unique match")},
       ["path", "old", "new"]),
    _t("run_command",
       "Run a shell command with the workspace as working directory and return its output and exit code. "
       "The user is asked to approve it. stdin is closed (no interactive prompts). Keep output small.",
       {"command": _S("The command line to run"),
        "shell": _S("powershell, cmd, bash or auto (default)"),
        "timeout": _I("Seconds before the command is killed")}, ["command"]),
    _t("run_python",
       "Run a Python script that exists in the workspace and return stdout+stderr and the exit code. Use it to "
       "test Python code you wrote. stdin is closed, so input() will fail - use arguments instead.",
       {"path": _S("Script path relative to the workspace"),
        "args": _L("Command-line arguments")}, ["path"]),
    _t("compile_c",
       "Compile C or C++ source files with gcc/g++ (chosen automatically from the file extensions) and return "
       "compiler errors/warnings. Set run_after=true to run the executable straight away. ALWAYS use this for "
       "C/C++ instead of calling gcc via run_command.",
       {"sources": _L("Source files relative to the workspace, e.g. ['main.c']"),
        "output": _S("Output executable name (optional)"),
        "flags": _L("Extra compiler/linker flags, e.g. ['-O2', '-lm']"),
        "compiler": _S("gcc, g++, clang or clang++ (optional, auto by default)"),
        "run_after": _B("Run the program after a successful compile")}, ["sources"]),
    _t("web_research",
       "BEST FIRST CHOICE for any question that needs facts from the web (docs, APIs, versions, errors, news). "
       "One call searches, opens the top pages and returns only the passages relevant to your query, plus the "
       "source URLs. Write the query as 3-8 keywords with exact names (library, function, error text, version, "
       "year), not as a sentence.",
       {"query": _S("Keyword query, e.g. 'pathlib Path.rglob python 3.12'"),
        "max_pages": _I("How many top results to read (1-4, default 3)")}, ["query"]),
    _t("web_search",
       "Search the web and list results (title, URL, snippet) WITHOUT reading the pages. Use it to find the "
       "right page, then call fetch_url on the best result. Prefer web_research when you just need an answer.",
       {"query": _S("Keyword query (3-8 words, exact names)"),
        "max_results": _I("Number of results, default 8")}, ["query"]),
    _t("fetch_url",
       "Download one web page / raw file / API response and return it as clean text. For long pages use "
       "find='keywords' to jump to the relevant parts, or page=2,3... to continue. GitHub 'blob' links are "
       "converted to raw files; Stack Overflow questions return the question plus top answers.",
       {"url": _S("Full https:// URL"),
        "find": _S("Optional keywords - returns only the matching passages of the page"),
        "page": _I("Page number for long documents (default 1)")}, ["url"]),
]
WEB_TOOLS = {"web_research", "web_search", "fetch_url"}

# Models sometimes use slightly wrong argument names - map them to the real ones.
ALIASES = {
    "q": "query", "search": "query", "search_query": "query", "keywords": "query", "text_query": "query",
    "file": "path", "filename": "path", "file_path": "path", "filepath": "path", "dir": "path", "directory": "path",
    "cmd": "command", "script": "path",
    "text": "content", "contents": "content", "file_content": "content", "code": "content", "body": "content",
    "old_str": "old", "old_text": "old", "new_str": "new", "new_text": "new",
    "link": "url", "uri": "url", "keyword": "find", "search_term": "find",
    "files": "sources", "source": "sources", "source_files": "sources",
    "arguments": "args", "limit": "max_results", "pages": "max_pages",
}

DOC_HINTS = {
    "docs.python.org": "Python: https://docs.python.org/3/library/<module>.html (e.g. .../library/pathlib.html), "
                       "language reference /3/reference/, tutorial /3/tutorial/, HOWTOs /3/howto/",
    "cppreference.com": "C/C++: https://en.cppreference.com/w/cpp/<area>/<name> e.g. /w/cpp/container/vector, "
                        "https://en.cppreference.com/w/c/<area>/<name> e.g. /w/c/string/byte/strlen",
    "lua.org": "Lua: https://www.lua.org/manual/5.4/manual.html (long page - use find='string.format')",
    "gcc.gnu.org": "GCC: https://gcc.gnu.org/onlinedocs/gcc/Option-Summary.html, .../gcc/Warning-Options.html, "
                   ".../gcc/Optimize-Options.html",
    "developer.mozilla.org": "MDN: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/<Name>, "
                             ".../Web/API/<Interface>, .../Web/CSS/<property>",
    "learn.microsoft.com": "Microsoft: https://learn.microsoft.com/en-us/powershell/ , /en-us/windows/win32/ , /en-us/cpp/",
    "github.com": "GitHub: raw file https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>; "
                  "repo page https://github.com/<owner>/<repo> (README)",
    "pypi.org": "PyPI: https://pypi.org/pypi/<package>/json gives latest version, dependencies and links",
    "stackoverflow.com": "Stack Overflow: web_search first, then fetch_url the question URL (returns question + top answers)",
}

LANG = {".py": "Python", ".c": "C", ".h": "C/C++ header", ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
        ".hpp": "C++ header", ".hxx": "C++ header",
        ".lua": "Lua", ".js": "JavaScript", ".ts": "TypeScript", ".java": "Java", ".cs": "C#", ".go": "Go",
        ".rs": "Rust", ".sh": "Shell", ".ps1": "PowerShell", ".bat": "Batch", ".json": "JSON",
        ".md": "Markdown", ".html": "HTML", ".css": "CSS", ".txt": "Text"}


class ToolError(Exception):
    pass


# ----------------------------------------------------------------------------
# HTML -> readable text
# ----------------------------------------------------------------------------
class ReadableHTML(HTMLParser):
    """Turns a web page into compact markdown-ish text (headings, lists, code blocks kept)."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
            "source", "track", "wbr"}
    BLOCKED = {"script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside", "form", "button",
               "select", "textarea", "canvas", "video", "audio", "object", "template"}
    BLOCKS = {"p", "div", "li", "tr", "section", "article", "main", "pre", "blockquote", "ul", "ol",
              "table", "dt", "dd", "details", "summary", "figure", "figcaption"}
    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    NOISE_CLASSES = ("sidebar", "cookie", "breadcrumb", "navbar", "advert", "headerlink", "skip-link",
                     "sr-only", "visually-hidden", "modal", "popup", "newsletter")
    NOISE_IDS = ("sidebar", "cookie", "footer", "breadcrumb", "navbar", "nav-", "topnav", "site-header", "masthead")
    MAIN_IDS = ("content", "main-content", "mw-content-text", "readme", "main")

    def __init__(self, base_url=""):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts = []
        self.pref_parts = []
        self.title_parts = []
        self.links = []
        self.stack = []
        self.skip = 0
        self.pre = 0
        self.pref = 0
        self.in_title = False
        self.cur_link = None
        self.pre_buf = []
        self.pre_blocks = []

    def _out(self):
        return self.pref_parts if self.pref else self.parts

    def _add(self, s):
        self._out().append(s)

    def _noise(self, a):
        if a.get("hidden") is not None or a.get("aria-hidden") == "true":
            return True
        style = (a.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        if (a.get("role") or "") in ("navigation", "banner", "complementary", "contentinfo", "search"):
            return True
        cls = (a.get("class") or "").lower()
        if cls and any(k in cls for k in self.NOISE_CLASSES):
            return True
        ident = (a.get("id") or "").lower()
        return bool(ident) and any(k in ident for k in self.NOISE_IDS)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.VOID:
            if tag == "br" and not self.skip:
                self._add("\n")
            return
        skip_here = (self.skip > 0 or tag in self.BLOCKED or (tag == "header" and not self.pref)
                     or self._noise(a))
        is_pref = (not skip_here) and (tag in ("main", "article") or a.get("role") == "main"
                                       or (a.get("id") or "") in self.MAIN_IDS)
        is_pre = (not skip_here) and tag == "pre"
        entry = {"tag": tag, "skip": skip_here, "pref": is_pref, "pre": is_pre, "code": False}
        self.stack.append(entry)
        if skip_here:
            self.skip += 1
            return
        if is_pref:
            self.pref += 1
        if is_pre:
            self.pre += 1
            if self.pre == 1:
                self.pre_buf = []
        if tag == "title":
            self.in_title = True
        elif tag in self.HEADINGS:
            self._add("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._add("\n- ")
        elif tag in ("td", "th"):
            self._add(" | ")
        elif tag in self.BLOCKS:
            self._add("\n")
        elif tag == "code" and not self.pre:
            entry["code"] = True
            self._add("`")
        if tag == "a" and a.get("href"):
            self.cur_link = {"url": urllib.parse.urljoin(self.base_url, a["href"]), "text": ""}

    def _close(self, e):
        tag = e["tag"]
        if e["skip"]:
            self.skip -= 1
            return
        if tag == "title":
            self.in_title = False
        if e["code"]:
            self._add("`")
        if e["pre"]:
            self.pre -= 1
            if self.pre == 0:
                block = "".join(self.pre_buf).strip("\n")
                self.pre_buf = []
                if block.strip():
                    self.pre_blocks.append(block)
                    self._add("\n\n\x00PRE%d\x00\n\n" % (len(self.pre_blocks) - 1))
        if e["pref"]:
            self.pref -= 1
        if tag in self.HEADINGS:
            self._add("\n\n")
        elif tag in self.BLOCKS:
            self._add("\n")
        if tag == "a" and self.cur_link:
            link = self.cur_link
            self.cur_link = None
            if link["text"].strip():
                self.links.append(link)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        idx = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                idx = i
                break
        if idx is None:
            return
        while len(self.stack) > idx:
            self._close(self.stack.pop())

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title_parts.append(data)
            return
        if self.cur_link is not None:
            self.cur_link["text"] += data
        if self.pre:
            self.pre_buf.append(data)
            return
        if not data.strip():
            out = self._out()
            if out and not out[-1].endswith((" ", "\n")):
                out.append(" ")
            return
        self._add(data)

    def result(self):
        main = "".join(self.pref_parts)
        text = main if len(main.strip()) > 300 else main + "\n\n" + "".join(self.parts)
        text = re.sub(r"[ \t\r\f\v\u00a0]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\x00PRE(\d+)\x00",
                      lambda m: "```\n" + self.pre_blocks[int(m.group(1))] + "\n```", text)
        links, seen = [], set()
        for l in self.links:
            u = l["url"].split("#")[0]
            if u in seen or not u.startswith(("http://", "https://")):
                continue
            seen.add(u)
            links.append({"url": u, "text": " ".join(l["text"].split())[:120]})
        return {"title": " ".join("".join(self.title_parts).split()), "text": text.strip(), "links": links}


class SearchParser(HTMLParser):
    """Parses DuckDuckGo's html and lite result pages. Each result = title link + snippet."""

    VOID = {"br", "img", "hr", "input", "meta", "link", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._a = None
        self._a_depth = 0
        self._sn = None
        self._sn_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        a = dict(attrs)
        cls = set((a.get("class") or "").split())
        if self._a is not None:
            self._a_depth += 1
        elif tag == "a" and (cls & {"result__a", "result-link"}) and a.get("href"):
            self._a = {"url": a["href"], "title": "", "snippet": ""}
            self._a_depth = 1
        if self._sn is not None:
            self._sn_depth += 1
        elif cls & {"result__snippet", "result-snippet"}:
            self._sn = ""
            self._sn_depth = 1

    def handle_data(self, data):
        if self._a is not None:
            self._a["title"] += data
        if self._sn is not None:
            self._sn += data

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self._a is not None:
            self._a_depth -= 1
            if self._a_depth <= 0:
                self._a["title"] = " ".join(self._a["title"].split())
                if self._a["title"]:
                    self.results.append(self._a)
                self._a = None
        if self._sn is not None:
            self._sn_depth -= 1
            if self._sn_depth <= 0:
                if self.results and not self.results[-1]["snippet"]:
                    self.results[-1]["snippet"] = " ".join(self._sn.split())
                self._sn = None


# ----------------------------------------------------------------------------
# Toolbox
# ----------------------------------------------------------------------------
class Toolbox:
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/124.0.0.0 Safari/537.36")

    HOST_KEYWORDS = [
        (r"\bpython|\bpip\b|\bpy\b|pandas|numpy|asyncio|tkinter|django|flask|pytest|dataclass|stdlib|pathlib",
         ("docs.python.org", "pypi.org")),
        (r"\bc\+\+|\bcpp\b|\bstd::|\bstl\b|\bc11\b|\bc99\b|\blibc\b|\bmalloc|\bprintf|\bstrlen|cppreference",
         ("cppreference.com",)),
        (r"\blua\b|luajit|\bgmod\b|\bglua\b", ("lua.org",)),
        (r"\bgcc\b|\bg\+\+|linker|compiler flag|makefile", ("gcc.gnu.org",)),
        (r"javascript|typescript|\bjs\b|\bcss\b|\bhtml\b|\bdom\b|web api|\bfetch\b|canvas|webgl|\bnode\b",
         ("developer.mozilla.org",)),
        (r"error|exception|traceback|segfault|not working|how to|how do|undefined reference",
         ("stackoverflow.com",)),
        (r"github|\brepo\b|repository|library|source code|example|implementation|open source|readme",
         ("github.com",)),
        (r"powershell|windows|win32|winapi|\.net|c#|csharp|msvc", ("learn.microsoft.com",)),
    ]
    STOP = set("a an the and or of to in on for with is are was be do does did how what why when which who "
               "can i you it this that from by as at into use using get make example tutorial".split())

    def __init__(self, cfg, approve):
        self.cfg = cfg
        self.approve = approve
        self.proc = None
        self.cache = OrderedDict()

    # ---- basics -------------------------------------------------------------
    def root(self):
        r = Path(self.cfg["workspace"]).resolve()
        r.mkdir(parents=True, exist_ok=True)
        return r

    def path(self, p):
        root = self.root()
        full = (root / (p or ".")).resolve()
        if full != root and root not in full.parents:
            raise ToolError("Path is outside the workspace: %s" % p)
        return full

    def kill(self):
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    def schemas(self):
        if self.cfg["web_mode"] == "off":
            return [t for t in TOOLS if t["function"]["name"] not in WEB_TOOLS]
        return TOOLS

    @staticmethod
    def parse_args(args):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return args if isinstance(args, dict) else {}

    def call(self, name, args):
        args = self.parse_args(args)
        fn = getattr(self, "t_" + name, None)
        if not fn:
            return "ERROR: unknown tool '%s'. Available tools: %s" % (
                name, ", ".join(t["function"]["name"] for t in self.schemas()))
        if name in WEB_TOOLS and self.cfg["web_mode"] == "off":
            return ("ERROR: Web access is switched off by the user. Answer from your own knowledge and say "
                    "clearly that it may be out of date.")
        params = inspect.signature(fn).parameters
        clean = {}
        for k, v in args.items():
            k2 = k if k in params else ALIASES.get(k)
            if k2 in params and k2 not in clean:
                clean[k2] = v
        try:
            return str(fn(**clean))
        except ToolError as e:
            return "ERROR: %s" % e
        except TypeError as e:
            return "ERROR: bad arguments for %s: %s. Parameters: %s" % (name, e, ", ".join(params))
        except Exception as e:
            return "ERROR: %s: %s" % (type(e).__name__, e)

    # ---- date/time ----------------------------------------------------------
    def t_get_datetime(self):
        n = now_info()
        return ("Now: %(pretty)s\nTimezone: %(tz)s\nISO 8601: %(iso)s\nDate: %(date)s   Time: %(time)s\n"
                "Day of year: %(doy)s   ISO week: %(week)s   Unix time: %(epoch)s" % n)

    # ---- files --------------------------------------------------------------
    def t_list_files(self, path=".", recursive=False):
        base = self.path(path)
        if not base.is_dir():
            raise ToolError("Not a directory: %s. Use list_files('.') to see the workspace root." % path)

        def size(p):
            try:
                n = p.stat().st_size
            except OSError:
                return ""
            return " (%s)" % (("%.1f KB" % (n / 1024)) if n >= 1024 else "%d B" % n)

        out = []
        if _bool(recursive):
            for dp, dn, fn in os.walk(base):
                dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
                for f in sorted(fn):
                    p = Path(dp, f)
                    out.append(str(p.relative_to(self.root())).replace("\\", "/") + size(p))
                if len(out) > 300:
                    break
        else:
            for p in sorted(base.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                out.append(p.name + ("/" if p.is_dir() else size(p)))
        return "\n".join(out[:300]) or "(empty folder)"

    def t_read_file(self, path, start_line=1, end_line=None):
        p = self.path(path)
        if not p.is_file():
            raise ToolError("File not found: %s. Use list_files to check the name." % path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        s = max(1, _int(start_line, 1))
        e = _int(end_line, s + 299) if end_line else s + 299
        e = min(e, len(lines), s + 399)
        if s > max(len(lines), 1):
            raise ToolError("start_line %d is past the end of the file (%d lines)." % (s, len(lines)))
        body = "\n".join("%5d | %s" % (i, lines[i - 1]) for i in range(s, e + 1))
        more = "" if e >= len(lines) else "\n[more: call read_file again with start_line=%d]" % (e + 1)
        return clip("[%s: lines %d-%d of %d]\n%s%s" % (path, s, e, len(lines), body, more), 20000)

    @staticmethod
    def _walk(base):
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
            for f in sorted(fn):
                yield Path(dp, f)

    def t_search_files(self, pattern, path=".", glob="*", regex=False, ignore_case=True):
        if not pattern:
            raise ToolError("pattern is empty.")
        base = self.path(path)
        flags = re.I if _bool(ignore_case, True) else 0
        try:
            rx = re.compile(pattern if _bool(regex) else re.escape(pattern), flags)
        except re.error as e:
            raise ToolError("Bad regular expression: %s" % e)
        files = [base] if base.is_file() else self._walk(base)
        out, scanned = [], 0
        for f in files:
            if glob and glob != "*" and not fnmatch.fnmatch(f.name, glob):
                continue
            try:
                if f.stat().st_size > 1_500_000:
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:2048]:
                continue
            scanned += 1
            rel = str(f.relative_to(self.root())).replace("\\", "/")
            for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append("%s:%d: %s" % (rel, i, line.strip()[:200]))
                    if len(out) >= 80:
                        return "\n".join(out) + "\n[stopped at 80 matches - narrow the pattern, path or glob]"
        if not out:
            return "No matches for %r in %d files." % (pattern, scanned)
        return "\n".join(out)

    def t_analyse_file(self, path):
        p = self.path(path)
        if not p.is_file():
            raise ToolError("File not found: %s" % path)
        data = p.read_bytes()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        ext = p.suffix.lower()
        info = [
            "path: %s" % path,
            "size: %d bytes" % len(data),
            "language: %s" % LANG.get(ext, ext or "unknown"),
            "lines: %d (blank: %d)" % (len(lines), sum(1 for l in lines if not l.strip())),
            "line endings: %s" % ("CRLF" if b"\r\n" in data else "LF"),
            "longest line: %d chars" % (max((len(l) for l in lines), default=0)),
            "TODO/FIXME: %d" % len(re.findall(r"TODO|FIXME", text)),
        ]
        if ext == ".py":
            try:
                tree = ast.parse(text)
                classes = [(n.name, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                funcs = [(n.name, n.lineno) for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                imps = sorted({(a.name if isinstance(n, ast.Import) else (n.module or "")).split(".")[0]
                               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                               for a in (n.names[:1] if n.names else [])})
                info.append("syntax: OK")
                info.append("classes: " + (", ".join("%s@%d" % c for c in classes) or "-"))
                info.append("functions: " + (", ".join("%s@%d" % f for f in funcs[:60]) or "-"))
                info.append("imports: " + (", ".join(imps) or "-"))
            except SyntaxError as e:
                info.append("syntax: ERROR line %s: %s" % (e.lineno, e.msg))
        else:
            funcs = []
            for i, l in enumerate(lines, 1):
                m = None
                if ext == ".lua":
                    m = re.search(r"function\s+([\w\.:]+)\s*\(", l)
                elif ext in (".js", ".ts"):
                    m = re.search(r"(?:function|class)\s+(\w+)", l)
                elif ext in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".java", ".cs", ".go", ".rs"):
                    m = re.match(r"^\s*(?:[A-Za-z_][\w:<>\*&,\s]+?)\s+\**([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$", l)
                    if m and m.group(1) in ("if", "while", "for", "switch", "return", "else"):
                        m = None
                if m:
                    funcs.append("%s@%d" % (m.group(1), i))
            if funcs:
                info.append("functions: " + ", ".join(funcs[:60]))
        info.append("--- first lines ---")
        info.extend(lines[:15])
        return "\n".join(info)

    @staticmethod
    def _pycheck(p, text):
        if p.suffix.lower() != ".py":
            return ""
        try:
            ast.parse(text)
            return "\npython syntax: OK"
        except SyntaxError as e:
            return "\nWARNING python syntax error at line %s: %s - fix it before running." % (e.lineno, e.msg)

    def t_write_file(self, path, content):
        p = self.path(path)
        content = str(content)
        if self.cfg["confirm_writes"] and not self.approve("write", "Write %d chars to %s" % (len(content), path)):
            raise ToolError("User declined the write.")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
        return "%s %s (%d bytes, %d lines)%s" % (
            "Overwrote" if existed else "Created", path, len(content.encode("utf-8")),
            content.count("\n") + 1, self._pycheck(p, content))

    def t_edit_file(self, path, old, new, replace_all=False):
        p = self.path(path)
        if not p.is_file():
            raise ToolError("File not found: %s" % path)
        old, new = str(old), str(new)
        if not old:
            raise ToolError("'old' is empty. To create or replace a whole file use write_file.")
        text = p.read_text(encoding="utf-8", errors="replace")
        n = text.count(old)
        if n == 0:
            hint = ""
            first = next((l.strip() for l in old.splitlines() if l.strip()), "")
            if first:
                lines = text.splitlines()
                best = max(range(len(lines)),
                           key=lambda i: difflib.SequenceMatcher(None, first, lines[i].strip()).ratio(),
                           default=None)
                if best is not None and difflib.SequenceMatcher(None, first, lines[best].strip()).ratio() > 0.6:
                    hint = " Closest match is on line %d: %r. Use read_file around it and copy the text exactly." % (
                        best + 1, lines[best].strip()[:100])
            raise ToolError("'old' text not found (whitespace and indentation must match exactly).%s" % (
                hint or " Use read_file and copy it exactly."))
        if n > 1 and not _bool(replace_all):
            raise ToolError("'old' text occurs %d times; include more surrounding lines to make it unique, "
                            "or set replace_all=true." % n)
        if self.cfg["confirm_writes"] and not self.approve("write", "Edit %s" % path):
            raise ToolError("User declined the edit.")
        result = text.replace(old, new) if _bool(replace_all) else text.replace(old, new, 1)
        p.write_text(result, encoding="utf-8", newline="\n")
        return "Edited %s (%d replacement%s)%s" % (path, n if _bool(replace_all) else 1,
                                                  "s" if _bool(replace_all) and n > 1 else "",
                                                  self._pycheck(p, result))

    # ---- running things -----------------------------------------------------
    def _run(self, argv, timeout=None):
        timeout = _int(timeout, self.cfg["timeout"]) or self.cfg["timeout"]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(self.root()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                creationflags=flags)
        except FileNotFoundError:
            return "ERROR: executable not found: %s" % argv[0]
        try:
            out, _ = self.proc.communicate(timeout=timeout)
            code = self.proc.returncode
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, _ = self.proc.communicate()
            code = "TIMEOUT after %ss (the program may be waiting for input or looping)" % timeout
        finally:
            self.proc = None
        return "exit_code: %s\n%s" % (code, clip(out or "(no output)"))

    def t_run_command(self, command, shell="auto", timeout=None):
        shell = (shell or "auto").lower()
        if shell == "auto":
            shell = "powershell" if os.name == "nt" else "bash"
        if shell == "powershell":
            exe = shutil.which("powershell") or shutil.which("pwsh")
            if not exe:
                raise ToolError("PowerShell not found.")
            argv = [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
        elif shell == "cmd":
            if os.name != "nt":
                raise ToolError("cmd is only available on Windows.")
            argv = ["cmd", "/c", command]
        elif shell == "bash":
            exe = shutil.which("bash")
            if not exe:
                raise ToolError("bash not found.")
            argv = [exe, "-c", command]
        else:
            raise ToolError("Unknown shell '%s'. Use powershell, cmd, bash or auto." % shell)
        if not self.approve("command", "[%s] %s\n(in %s)" % (shell, command, self.root())):
            raise ToolError("User declined to run the command.")
        return self._run(argv, timeout)

    @staticmethod
    def _python_exe():
        if getattr(sys, "frozen", False):
            return shutil.which("python") or shutil.which("py") or "python"
        return sys.executable

    def t_run_python(self, path, args=None):
        p = self.path(path)
        if not p.is_file():
            raise ToolError("File not found: %s" % path)
        argl = _list(args)
        if not self.approve("command", "python %s %s" % (path, " ".join(argl))):
            raise ToolError("User declined to run the script.")
        return self._run([self._python_exe(), "-u", str(p)] + argl)

    def t_compile_c(self, sources, output=None, flags=None, compiler=None, run_after=False):
        srcs = _list(sources)
        if not srcs:
            raise ToolError("No source files given.")
        exts = set()
        for s in srcs:
            p = self.path(s)
            if not p.is_file():
                raise ToolError("Source not found: %s" % s)
            exts.add(p.suffix.lower())

        cpp_exts = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hxx"}
        is_cpp = bool(exts & cpp_exts)

        if compiler:
            compiler = str(compiler).lower().strip()
            if compiler not in ("gcc", "g++", "clang", "clang++", "gcc.exe", "g++.exe"):
                compiler = "g++" if is_cpp else "gcc"
        else:
            compiler = "g++" if is_cpp else "gcc"

        exe = shutil.which(compiler)
        if not exe:
            for alt in (["g++", "gcc"] if is_cpp else ["gcc", "g++"]):
                exe = shutil.which(alt)
                if exe:
                    compiler = alt
                    break
        if not exe:
            raise ToolError("Neither gcc nor g++ found in PATH. Add the MinGW bin folder to PATH.")

        if not output:
            output = Path(srcs[0]).stem + (".exe" if os.name == "nt" else "")
            if output in ("a.out", "a.exe"):
                output = "program" + (".exe" if os.name == "nt" else "")

        self.path(output)
        flag_list = _list(flags) or ["-Wall", "-Wextra", "-g"]
        argv = [exe] + srcs + ["-o", output] + flag_list

        if not self.approve("command", " ".join(argv)):
            raise ToolError("User declined to compile.")

        res = self._run(argv)
        if res.startswith("exit_code: 0") and _bool(run_after):
            target = str(self.path(output))
            if self.approve("command", "Run compiled program: %s" % output):
                res += "\n--- running %s ---\n%s" % (output, self._run([target]))
        return res

    # ---- web: safety --------------------------------------------------------
    def approved(self, url):
        mode = self.cfg["web_mode"]
        if mode == "open":
            return True
        if mode == "off":
            return False
        u = urllib.parse.urlparse(url)
        host = (u.hostname or "").lower().removeprefix("www.")
        for link in self.cfg["links"]:
            l = urllib.parse.urlparse(link if "://" in link else "https://" + link)
            allowed = (l.hostname or "").lower().removeprefix("www.")
            if not allowed:
                continue
            if host == allowed or host.endswith("." + allowed):
                if l.path in ("", "/") or u.path.startswith(l.path):
                    return True
        return False

    @staticmethod
    def _safe_host(url):
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        host = u.hostname.lower()
        if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
            return False
        try:
            infos = socket.getaddrinfo(host, None)
        except Exception:
            return True                       # DNS problems surface later as a fetch error
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0].split("%")[0])
            except ValueError:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True

    def _approved_hosts(self):
        hosts = []
        for link in self.cfg["links"]:
            h = (urllib.parse.urlparse(link if "://" in link else "https://" + link).hostname or "").lower()
            if h and h not in hosts:
                hosts.append(h)
        return hosts

    # ---- web: http ----------------------------------------------------------
    @classmethod
    def _get(cls, url, limit=2_000_000, data=None, extra=None, timeout=25):
        headers = {
            "User-Agent": cls.UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        if extra:
            headers.update(extra)
        body = urllib.parse.urlencode(data).encode("utf-8") if isinstance(data, dict) else data
        request = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ctype = response.headers.get_content_type() or ""
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(limit)
            final = response.geturl()
        if raw[:2] == b"\x1f\x8b":            # some APIs always gzip, even when told not to
            try:
                raw = zlib.decompressobj(31).decompress(raw)
            except Exception:
                pass
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
        return text, ctype.lower(), final

    @staticmethod
    def _ddg_url(href):
        href = html.unescape(href or "").strip()
        if not href:
            return ""
        full = urllib.parse.urljoin("https://duckduckgo.com/", href)
        p = urllib.parse.urlparse(full)
        if (p.hostname or "").endswith("duckduckgo.com"):
            q = urllib.parse.parse_qs(p.query)
            return q["uddg"][0] if "uddg" in q else ""      # no uddg = ad / internal link
        return full

    def _ddg(self, q):
        errors = []
        for endpoint in ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"):
            try:
                raw, _, _ = self._get(endpoint, 1_500_000, data={"q": q, "kl": "wt-wt"},
                                      extra={"Referer": endpoint})
                if "anomaly-modal" in raw or "bots use DuckDuckGo too" in raw:
                    errors.append("search engine asked for a captcha (rate limited)")
                    continue
                parser = SearchParser()
                parser.feed(raw)
                parser.close()
                if parser.results:
                    return parser.results
            except Exception as e:
                errors.append(str(e))
        if errors:
            raise RuntimeError("; ".join(errors[:2]))
        return []

    def _host_ok(self, host):
        return self.approved("https://%s/" % host)

    def _rank_hosts(self, q):
        ql = q.lower()
        common = {"www", "com", "org", "net", "io", "dev", "gnu", "en", "docs", "developer", "learn", "api", "raw",
                  "githubusercontent"}
        scored = []
        for order, h in enumerate(self._approved_hosts()):
            if h.startswith(("raw.", "api.")):
                continue
            sc = 0
            for tok in h.split("."):
                if tok not in common and len(tok) > 2 and tok in ql:
                    sc += 4
            for rx, targets in self.HOST_KEYWORDS:
                if re.search(rx, ql) and any(t in h for t in targets):
                    sc += 2
            scored.append((-sc, order, h))
        scored.sort()
        ranked = [h for s, o, h in scored if s < 0]
        if not ranked:
            ranked = [h for s, o, h in scored if "stackoverflow" in h or "github" in h] or [h for s, o, h in scored[:2]]
        return ranked

    def _api_fallbacks(self, q, seen):
        """Structured sources used when the search engine returns little."""
        out = []
        if self._host_ok("github.com"):
            try:
                raw, _, _ = self._get("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                    {"q": q, "per_page": 5}), extra={"Accept": "application/vnd.github+json"}, timeout=15)
                for it in json.loads(raw).get("items", [])[:5]:
                    url = it.get("html_url", "")
                    if url and url not in seen:
                        out.append({"title": "GitHub repo: %s (%s stars)" % (it.get("full_name"), it.get("stargazers_count")),
                                    "url": url, "snippet": it.get("description") or ""})
            except Exception:
                pass
        if self._host_ok("stackoverflow.com"):
            try:
                raw, _, _ = self._get("https://api.stackexchange.com/2.3/search/advanced?" + urllib.parse.urlencode(
                    {"order": "desc", "sort": "relevance", "q": q, "site": "stackoverflow", "pagesize": 5}), timeout=15)
                for it in json.loads(raw).get("items", [])[:5]:
                    url = it.get("link", "")
                    if url and url not in seen:
                        out.append({"title": "Stack Overflow: " + html.unescape(it.get("title", "")), "url": url,
                                    "snippet": "score %s, %s answer(s)%s" % (
                                        it.get("score"), it.get("answer_count"),
                                        ", has accepted answer" if it.get("is_answered") else "")})
            except Exception:
                pass
        return out

    def _search(self, query, n=8):
        q = " ".join(str(query).split())
        queries = []
        if self.cfg["web_mode"] == "approved":
            queries += ["site:%s %s" % (h, q) for h in self._rank_hosts(q)[:3]]
        queries.append(q)
        batches, errors = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries))) as ex:
            futures = [ex.submit(self._ddg, x) for x in queries]
            for f in futures:
                try:
                    batches.append(f.result())
                except Exception as e:
                    errors.append(str(e))
                    batches.append([])
        merged, seen = [], set()
        for i in range(max((len(b) for b in batches), default=0)):
            for b in batches:
                if i >= len(b):
                    continue
                url = self._ddg_url(b[i]["url"])
                key = url.split("#")[0].rstrip("/").lower()
                if not url.startswith(("http://", "https://")) or key in seen or not self.approved(url):
                    continue
                seen.add(key)
                merged.append({"title": b[i]["title"], "url": url, "snippet": b[i]["snippet"]})
        if len(merged) < 3:
            for it in self._api_fallbacks(q, seen):
                merged.append(it)
                seen.add(it["url"].rstrip("/").lower())
        return merged[:n], sorted(set(errors))

    # ---- web: text helpers --------------------------------------------------
    @classmethod
    def _terms(cls, s):
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_.:+#\-]*|\d+(?:\.\d+)+", str(s).lower())
        out = []
        for w in words:
            w = w.strip(".:-")
            if len(w) < 2 or w in cls.STOP:
                continue
            for x in [w] + [p for p in re.split(r"[.:]+", w) if len(p) >= 3 and p != w]:
                if x not in out and x not in cls.STOP:
                    out.append(x)
        return out[:14]

    @staticmethod
    def _blocks(text):
        blocks = []
        for part in re.split(r"(```.*?```)", text, flags=re.S):
            if part.startswith("```"):
                blocks.append(part)
                continue
            for b in re.split(r"\n\s*\n", part):
                b = b.strip()
                while len(b) > 1400:
                    cut = b.rfind("\n", 300, 1400)
                    cut = cut if cut > 0 else 1400
                    blocks.append(b[:cut].strip())
                    b = b[cut:].strip()
                if b:
                    blocks.append(b)
        return blocks

    @classmethod
    def _passages(cls, text, terms, budget=3500):
        """Return (text, hits): the parts of `text` that best match `terms`, in original order."""
        blocks = cls._blocks(text)
        if not blocks:
            return "", 0
        if not terms:
            return text[:budget], 0
        low = [b.lower() for b in blocks]
        heading_of, cur = [], ""
        for b in blocks:
            if b.startswith("#"):
                cur = b.splitlines()[0]
            heading_of.append(cur)
        base = []
        for i, b in enumerate(low):
            sc = 0.0
            for t in terms:
                c = b.count(t)
                if c:
                    sc += 1.0 + min(c, 4) * 0.25
            if blocks[i].startswith("#"):
                sc *= 1.5
            base.append(sc)
        scores = list(base)
        for i, b in enumerate(blocks):
            if b.startswith("```") and i > 0:
                scores[i] += 0.6 * base[i - 1]
            if i + 1 < len(blocks) and blocks[i + 1].startswith("```"):
                scores[i] += 0.3 * base[i + 1]
        order = sorted(range(len(blocks)), key=lambda i: -scores[i])
        chosen, used = [], 0
        for i in order:
            if scores[i] <= 0:
                break
            size = min(len(blocks[i]), 2200) + 40
            if used + size > budget and chosen:
                continue
            chosen.append(i)
            used += size
        if not chosen:
            return text[:budget], 0
        chosen.sort()
        out, last_head, prev = [], "", None
        for i in chosen:
            piece = blocks[i][:2200]
            if prev is not None and i != prev + 1:
                out.append("[...]")
            if heading_of[i] and heading_of[i] != last_head and not blocks[i].startswith("#"):
                out.append(heading_of[i])
            last_head = heading_of[i] or last_head
            out.append(piece)
            prev = i
        return "\n\n".join(out), len(chosen)

    @staticmethod
    def _outline(text, maxn=30):
        out, in_code = [], False
        for line in text.splitlines():
            if line.startswith("```"):
                in_code = not in_code
            elif not in_code and line.startswith("#"):
                out.append(line.strip()[:100])
                if len(out) >= maxn:
                    break
        return out

    # ---- web: loading pages -------------------------------------------------
    @staticmethod
    def _normalize_url(url):
        url = str(url).strip().strip("<>\"'")
        if "://" not in url:
            url = "https://" + url
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
        if m:
            url = "https://raw.githubusercontent.com/%s/%s/%s" % m.groups()
        return url.split("#")[0]

    @staticmethod
    def _pypi_summary(data):
        info = data.get("info") or {}
        urls = data.get("urls") or []
        lines = ["PACKAGE: %s %s" % (info.get("name"), info.get("version")),
                 "SUMMARY: %s" % info.get("summary"),
                 "REQUIRES-PYTHON: %s" % info.get("requires_python"),
                 "HOME PAGE: %s" % (info.get("home_page") or ""),
                 "PROJECT URLS: %s" % json.dumps(info.get("project_urls") or {}, ensure_ascii=False)]
        if urls:
            lines.append("LATEST FILES UPLOADED: %s" % urls[0].get("upload_time", ""))
        lines.append("DEPENDENCIES: %s" % "; ".join((info.get("requires_dist") or [])[:25]))
        lines.append("\nDESCRIPTION (start):\n" + (info.get("description") or "")[:3500])
        return "\n".join(lines)

    def _load_stackoverflow(self, qid):
        base = "https://api.stackexchange.com/2.3/questions/%s" % qid
        raw, _, _ = self._get(base + "?site=stackoverflow&filter=withbody", timeout=20)
        q = (json.loads(raw).get("items") or [None])[0]
        if not q:
            raise RuntimeError("question not found")
        raw, _, _ = self._get(base + "/answers?" + urllib.parse.urlencode(
            {"order": "desc", "sort": "votes", "site": "stackoverflow", "filter": "withbody", "pagesize": 4}), timeout=20)
        answers = json.loads(raw).get("items") or []

        def body(h):
            p = ReadableHTML()
            p.feed(h or "")
            p.close()
            return p.result()["text"]

        title = html.unescape(q.get("title", ""))
        parts = ["# %s" % title, "QUESTION (score %s, %s answers):\n%s" % (
            q.get("score"), q.get("answer_count"), body(q.get("body")))]
        for i, a in enumerate(answers, 1):
            parts.append("## ANSWER %d (score %s%s)\n%s" % (
                i, a.get("score"), ", ACCEPTED" if a.get("is_accepted") else "", body(a.get("body"))))
        return {"title": title, "text": "\n\n".join(parts), "links": []}

    def _load(self, url):
        url = self._normalize_url(url)
        if url in self.cache:
            self.cache.move_to_end(url)
            return self.cache[url]
        if not self._safe_host(url):
            raise ToolError("Only public http/https URLs are allowed.")
        if not self.approved(url):
            raise ToolError("This site is not on the approved list. Pick a result from web_search, use an "
                            "approved site (%s), or ask the user to add it under Links." % ", ".join(self._approved_hosts()[:8]))
        page = None
        m = re.match(r"https?://(?:www\.)?stackoverflow\.com/questions/(\d+)", url)
        if m and self._host_ok("stackoverflow.com"):
            try:
                page = self._load_stackoverflow(m.group(1))
            except Exception:
                page = None
        if page is None:
            try:
                body, ctype, final = self._get(url)
            except urllib.error.HTTPError as e:
                hint = " - check the URL (use web_search to find the right page)" if e.code in (404, 410) else ""
                raise ToolError("HTTP %s for %s%s" % (e.code, url, hint))
            except Exception as e:
                raise ToolError("Fetch failed for %s: %s" % (url, e))
            if final.split("#")[0] != url and not self.approved(final):
                raise ToolError("Redirected to a site that is not approved: %s" % final)
            if "html" in ctype or body.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
                parser = ReadableHTML(final)
                try:
                    parser.feed(body)
                    parser.close()
                    r = parser.result()
                except Exception:
                    plain = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", body)
                    plain = re.sub(r"(?is)<[^>]+>", " ", plain)
                    r = {"title": "", "text": re.sub(r"\s+", " ", plain)[:40000], "links": []}
                page = {"title": r["title"], "text": r["text"] or "(no readable text extracted)", "links": r["links"]}
            elif "json" in ctype:
                try:
                    data = json.loads(body)
                    if re.match(r"https?://pypi\.org/pypi/[^/]+(/[^/]+)?/json", url):
                        text = self._pypi_summary(data)
                    else:
                        text = json.dumps(data, indent=2, ensure_ascii=False)
                except Exception:
                    text = body
                page = {"title": "", "text": text, "links": []}
            else:
                page = {"title": "", "text": body, "links": []}
        page["url"] = url
        self.cache[url] = page
        while len(self.cache) > 24:
            self.cache.popitem(last=False)
        return page

    # ---- web: tools ---------------------------------------------------------
    def _no_results(self, query, errors):
        lines = ["NO RESULTS for: %s" % query]
        if errors:
            lines.append("Search backend problem: %s" % "; ".join(errors))
        lines += [
            "",
            "What to do next (do NOT repeat the same query):",
            "- Rephrase with 3-6 different keywords: exact function/library/error names, drop filler words and versions.",
            "- Try a query aimed at one approved site, e.g. 'pathlib glob' for the Python docs.",
            "- Open an official page directly with fetch_url. Known patterns:",
        ]
        shown = 0
        for key, hint in DOC_HINTS.items():
            if self.cfg["web_mode"] == "open" or any(key in h for h in self._approved_hosts()):
                lines.append("    * " + hint)
                shown += 1
        if not shown:
            lines.append("    * (no approved sites configured - ask the user to add some under Links)")
        return "\n".join(lines)

    def t_web_search(self, query, max_results=8):
        if not str(query).strip():
            raise ToolError("query is empty.")
        results, errors = self._search(query, max(1, min(_int(max_results, 8), 12)))
        if not results:
            return self._no_results(query, errors)
        lines = ["WEB SEARCH: %s   (searched %s, %s)" % (
            query, now_info()["date"], "approved sites only" if self.cfg["web_mode"] == "approved" else "open web")]
        for i, r in enumerate(results, 1):
            lines.append("\n%d. %s\n   URL: %s\n   %s" % (i, r["title"], r["url"], r["snippet"][:300]))
        lines.append("\nNEXT STEP: fetch_url(url, find='keywords') on the most relevant result, or "
                     "web_research(query) to read the top pages automatically.")
        return "\n".join(lines)

    def t_web_research(self, query, max_pages=3):
        if not str(query).strip():
            raise ToolError("query is empty.")
        n = max(1, min(_int(max_pages, 3), 4))
        results, errors = self._search(query, 10)
        if not results:
            return self._no_results(query, errors)
        terms = self._terms(query)
        picks, per_host = [], {}
        for r in results:
            host = urllib.parse.urlparse(r["url"]).hostname or ""
            if per_host.get(host, 0) >= 2:
                continue
            per_host[host] = per_host.get(host, 0) + 1
            picks.append(r)
            if len(picks) >= n:
                break

        def work(item):
            try:
                page = self._load(item["url"])
                text, hits = self._passages(page["text"], terms, 3600)
                return item, page, text, hits, None
            except Exception as e:
                return item, None, "", 0, str(e).replace("ERROR: ", "")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(picks)) as ex:
            done = list(ex.map(work, picks))

        out = ["WEB RESEARCH: %s   (today is %s; %s)" % (query, now_info()["date"],
               "approved sites only" if self.cfg["web_mode"] == "approved" else "open web"), ""]
        read_urls = set()
        for i, (item, page, text, hits, err) in enumerate(done, 1):
            read_urls.add(item["url"])
            title = (page or {}).get("title") or item["title"]
            out.append("=== [%d] %s\nURL: %s" % (i, title, item["url"]))
            if err:
                out.append("(could not read this page: %s)\n" % err)
                continue
            if not hits:
                out.append("(no passage matched your keywords closely - showing the start of the page)")
            out.append(text.strip() + "\n")
        rest = [r for r in results if r["url"] not in read_urls]
        if rest:
            out.append("NOT READ (use fetch_url if you need them):")
            out += ["- %s - %s" % (r["title"][:90], r["url"]) for r in rest[:6]]
        out.append("\nTIP: for more detail on one page call fetch_url(url, find='specific keywords'). "
                   "Cite the URLs you used under 'Sources:'.")
        return clip("\n".join(out), 15000)

    def t_fetch_url(self, url, find=None, page=1):
        pg = self._load(url)
        text, title = pg["text"], pg.get("title") or ""
        head = "[%s]%s" % (pg["url"], ("  " + title) if title else "")
        if find and str(find).strip():
            terms = self._terms(find)
            body, hits = self._passages(text, terms, 7500)
            link_hits = [l for l in pg["links"] if any(t in (l["text"] + " " + l["url"]).lower() for t in terms)][:8]
            if not hits:
                return clip("%s\nNo passage matches %r on this page (%d chars). Page outline:\n%s\n\n"
                            "Try different keywords, or fetch_url(url) without find to read from the top." % (
                                head, find, len(text), "\n".join(self._outline(text)) or "(no headings)"), 12000)
            extra = ""
            if link_hits:
                extra = "\n\nRELATED LINKS ON THIS PAGE:\n" + "\n".join("%s -> %s" % (l["text"], l["url"]) for l in link_hits)
            return "%s\nPASSAGES matching %r:\n\n%s%s" % (head, find, body, extra)
        size = 10000
        total = max(1, math.ceil(len(text) / size))
        p = max(1, min(_int(page, 1), total))
        chunk = text[(p - 1) * size: p * size]
        lines = [head, "PAGE %d of %d  (%d chars total)" % (p, total, len(text))]
        if p == 1 and total > 1:
            outline = self._outline(text)
            if outline:
                lines.append("OUTLINE:\n" + "\n".join(outline))
        lines += ["", chunk]
        if p < total:
            lines.append("\n[continues - fetch_url(url, page=%d) or fetch_url(url, find='keywords')]" % (p + 1))
        elif pg["links"]:
            lines.append("\nLINKS ON THIS PAGE:\n" + "\n".join("%s -> %s" % (l["text"], l["url"]) for l in pg["links"][:25]))
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Theme + small widgets
# ----------------------------------------------------------------------------
T = {
    "bg": "#0d1117", "panel": "#12161d", "card": "#171c25", "card2": "#1f2631", "input": "#151a22",
    "border": "#262e3a", "text": "#e6edf3", "dim": "#8b95a5", "faint": "#5b6574",
    "accent": "#7c9cff", "accent_hover": "#9bb3ff", "accent_fg": "#0b1020", "sel": "#24304a",
    "good": "#3fb950", "warn": "#e3b341", "bad": "#f85149",
    "code_bg": "#0a0d12", "code_fg": "#c9d6e8", "think": "#7d8696",
}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def pick_font(root, candidates, default):
    try:
        fams = set(tkfont.families(root))
    except Exception:
        return default
    for c in candidates:
        if c in fams:
            return c
    return default


class FlatButton(tk.Label):
    def __init__(self, parent, text, command=None, kind="ghost", font=None, padx=12, pady=6):
        super().__init__(parent, text=text, font=font, padx=padx, pady=pady, bd=0)
        self.command, self.kind, self.enabled, self.hover = command, kind, True, False
        self.base_bg = parent.cget("bg")
        self._paint()
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._click)

    def _paint(self):
        if not self.enabled:
            bg = T["card2"] if self.kind != "ghost" else self.base_bg
            fg = T["faint"]
        elif self.kind == "primary":
            bg, fg = (T["accent_hover"] if self.hover else T["accent"]), T["accent_fg"]
        elif self.kind == "danger":
            bg, fg = ("#ff7b72" if self.hover else T["bad"]), "#1a0b0a"
        elif self.kind == "soft":
            bg, fg = (T["border"] if self.hover else T["card2"]), T["text"]
        else:
            bg, fg = (T["card2"] if self.hover else self.base_bg), (T["text"] if self.hover else T["dim"])
        self.configure(bg=bg, fg=fg, cursor="hand2" if self.enabled else "arrow")

    def _set_hover(self, h):
        self.hover = h
        self._paint()

    def _click(self, e):
        if self.enabled and self.command and 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            self.command()

    def set_enabled(self, flag):
        self.enabled = bool(flag)
        self.hover = False
        self._paint()


class Toggle(tk.Canvas):
    def __init__(self, parent, text, variable, command=None, font=None):
        self.var, self.command, self.label, self.font = variable, command, text, font
        width = font.measure(text) + 50
        super().__init__(parent, width=width, height=26, bg=parent.cget("bg"), highlightthickness=0, cursor="hand2")
        self.bind("<Button-1>", self._toggle)
        self.var.trace_add("write", lambda *a: self._draw())
        self._draw()

    def _toggle(self, _=None):
        self.var.set(not self.var.get())
        if self.command:
            self.command()

    def _draw(self):
        self.delete("all")
        on = bool(self.var.get())
        track = T["accent"] if on else T["border"]
        self.create_oval(2, 5, 18, 21, fill=track, outline=track)
        self.create_oval(16, 5, 32, 21, fill=track, outline=track)
        self.create_rectangle(10, 5, 24, 21, fill=track, outline=track)
        kx = 21 if on else 4
        self.create_oval(kx, 7, kx + 12, 19, fill="#ffffff" if on else T["dim"], outline="")
        self.create_text(40, 13, text=self.label, anchor="w", fill=T["text"] if on else T["dim"], font=self.font)


class Segmented(tk.Frame):
    def __init__(self, parent, options, variable, command=None, font=None):
        super().__init__(parent, bg=T["card2"], padx=2, pady=2)
        self.var, self.command, self.items = variable, command, {}
        for value, label in options:
            lb = tk.Label(self, text=label, font=font, padx=9, pady=2, cursor="hand2", bd=0)
            lb.pack(side="left")
            lb.bind("<Button-1>", lambda e, v=value: self._pick(v))
            self.items[value] = lb
        self.var.trace_add("write", lambda *a: self._paint())
        self._paint()

    def _pick(self, v):
        self.var.set(v)
        if self.command:
            self.command()

    def _paint(self):
        cur = self.var.get()
        for v, lb in self.items.items():
            if v == cur:
                lb.configure(bg=T["accent"], fg=T["accent_fg"])
            else:
                lb.configure(bg=T["card2"], fg=T["dim"])


# ----------------------------------------------------------------------------
# Markdown streaming (pure logic - no tkinter, so it is easy to test)
# ----------------------------------------------------------------------------
INLINE_RE = re.compile(r"(`[^`\n]+`|\*\*[^*\n]+\*\*|(?<![\w*])\*[^*\s][^*\n]*?\*(?![\w*]))")


class MarkdownStream:
    CODE_WIDTH = 78

    def __init__(self):
        self.buf = ""
        self.in_code = False

    def feed(self, chunk):
        """Add text; returns display ops [(text, (tags...))] for every COMPLETE line."""
        self.buf += chunk
        ops = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            ops += self.line(line.rstrip("\r"))
        return ops

    def partial(self):
        if not self.buf:
            return None
        return self.buf, (("codeblock",) if self.in_code else ("assistant",))

    def flush(self):
        ops = self.line(self.buf.rstrip("\r")) if self.buf else []
        self.buf, self.in_code = "", False
        return ops

    def inline(self, text, base):
        ops = []
        for part in INLINE_RE.split(text):
            if not part:
                continue
            if part.startswith("`") and part.endswith("`") and len(part) > 2:
                ops.append((part[1:-1], base + ("inlinecode",)))
            elif part.startswith("**") and part.endswith("**") and len(part) > 4:
                ops.append((part[2:-2], base + ("bold",)))
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                ops.append((part[1:-1], base + ("italic",)))
            else:
                ops.append((part, base))
        return ops

    def line(self, s):
        stripped = s.strip()
        if stripped.startswith("```"):
            if not self.in_code:
                self.in_code = True
                lang = stripped[3:].strip()
                label = (" %s" % lang.upper()) if lang else ""
                return [(label.ljust(self.CODE_WIDTH) + "\n", ("codelang",))]
            self.in_code = False
            return [(" " * self.CODE_WIDTH + "\n", ("codeblock",)), ("\n", ("small",))]
        if self.in_code:
            return [(" " + s.expandtabs(4).ljust(self.CODE_WIDTH) + "\n", ("codeblock",))]
        if not stripped:
            return [("\n", ("small",))]
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            return [(m.group(2).replace("`", "") + "\n", ("h%d" % len(m.group(1)),))]
        if re.match(r"^\s*([-*_])\1{2,}\s*$", s):
            return [("─" * 40 + "\n", ("hr",))]
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", s)
        if m:
            marker = "•" if m.group(2) in "-*+" else m.group(2)
            pad = "    " * (len(m.group(1).expandtabs(4)) // 4)
            return ([(pad + marker + "  ", ("bullet",))] + self.inline(m.group(3), ("bullet",))
                    + [("\n", ("bullet",))])
        if stripped.startswith(">"):
            return self.inline(stripped.lstrip("> ").strip(), ("quote",)) + [("\n", ("quote",))]
        if stripped.startswith("|") and stripped.endswith("|"):
            return [(s + "\n", ("table",))]
        return self.inline(s, ("assistant",)) + [("\n", ("assistant",))]


def ui_clip(text, max_lines=14, max_chars=1500):
    lines = text.splitlines()
    extra = ""
    if len(lines) > max_lines:
        extra = "\n… %d more lines" % (len(lines) - max_lines)
        lines = lines[:max_lines]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out, extra = out[:max_chars], "\n… (shortened)"
    return out + extra


class ChatView(tk.Frame):
    def __init__(self, parent, fonts):
        super().__init__(parent, bg=T["bg"])
        self.f = fonts
        self.md = MarkdownStream()
        self.kind = None
        self.partial = False
        self.text = tk.Text(self, wrap="word", bg=T["bg"], fg=T["text"], insertbackground=T["text"],
                            relief="flat", bd=0, padx=26, pady=14, highlightthickness=0,
                            selectbackground=T["sel"], selectforeground=T["text"], state="disabled",
                            cursor="arrow", spacing1=1, spacing3=3)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview, style="Dark.Vertical.TScrollbar")
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self._tags()
        menu = tk.Menu(self, tearoff=0, bg=T["card"], fg=T["text"], activebackground=T["accent"],
                       activeforeground=T["accent_fg"], bd=0)
        menu.add_command(label="Copy", command=lambda: self.text.event_generate("<<Copy>>"))
        menu.add_command(label="Select all", command=lambda: self.text.tag_add("sel", "1.0", "end"))
        self.text.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        self.text.bind("<Control-a>", lambda e: (self.text.tag_add("sel", "1.0", "end"), "break")[1])

    def _tags(self):
        t, f = self.text, self.f
        t.tag_configure("assistant", font=f["ui"], foreground=T["text"], lmargin1=2, lmargin2=2, spacing3=4)
        t.tag_configure("small", font=(f["ui_family"], 4))
        t.tag_configure("h1", font=f["h1"], foreground=T["text"], spacing1=10, spacing3=4)
        t.tag_configure("h2", font=f["h2"], foreground=T["text"], spacing1=8, spacing3=3)
        t.tag_configure("h3", font=f["h3"], foreground=T["accent"], spacing1=6, spacing3=2)
        t.tag_configure("h4", font=f["ui_b"], foreground=T["dim"], spacing1=4)
        t.tag_configure("bullet", font=f["ui"], foreground=T["text"], lmargin1=14, lmargin2=34, spacing3=2)
        t.tag_configure("quote", font=f["ui_i"], foreground=T["dim"], lmargin1=14, lmargin2=14)
        t.tag_configure("hr", foreground=T["border"])
        t.tag_configure("table", font=f["mono"], foreground=T["code_fg"], lmargin1=6, lmargin2=6)
        t.tag_configure("codelang", font=f["mono_s"], foreground=T["faint"], background=T["code_bg"],
                        lmargin1=6, lmargin2=6, spacing1=6, spacing3=0)
        t.tag_configure("codeblock", font=f["mono"], foreground=T["code_fg"], background=T["code_bg"],
                        lmargin1=6, lmargin2=6, spacing1=0, spacing3=0)
        t.tag_configure("bold", font=f["ui_b"])
        t.tag_configure("italic", font=f["ui_i"])
        t.tag_configure("inlinecode", font=f["mono"], foreground="#ffa657", background=T["card2"])
        t.tag_configure("user_label", font=f["label"], foreground=T["accent"], spacing1=18, spacing3=3)
        t.tag_configure("user", font=f["ui"], foreground=T["text"], background=T["card"], lmargin1=10,
                        lmargin2=10, rmargin=10, spacing1=3, spacing3=3)
        t.tag_configure("asst_label", font=f["label"], foreground=T["good"], spacing1=18, spacing3=3)
        t.tag_configure("think_label", font=f["label"], foreground=T["faint"], spacing1=10, spacing3=2)
        t.tag_configure("think", font=f["ui_i"], foreground=T["think"], lmargin1=12, lmargin2=12, spacing3=1)
        t.tag_configure("tool_head", font=f["mono_b"], foreground=T["warn"], spacing1=10, spacing3=2)
        t.tag_configure("tool_body", font=f["mono"], foreground=T["code_fg"], background=T["code_bg"],
                        lmargin1=14, lmargin2=14, spacing3=0)
        t.tag_configure("result_ok", font=f["mono_s"], foreground=T["dim"], lmargin1=14, lmargin2=14, spacing3=2)
        t.tag_configure("result_err", font=f["mono_s"], foreground=T["bad"], lmargin1=14, lmargin2=14, spacing3=2)
        t.tag_configure("info", font=f["ui_s"], foreground=T["faint"], spacing1=6)
        t.tag_configure("error", font=f["ui_b"], foreground=T["bad"], spacing1=8, spacing3=4)
        t.tag_configure("welcome_title", font=f["h1"], foreground=T["text"], spacing1=40, spacing3=4, justify="center")
        t.tag_configure("welcome_sub", font=f["ui"], foreground=T["dim"], spacing3=18, justify="center")
        t.tag_configure("chip_row", justify="center", spacing1=4, spacing3=4)
        # later tags win: keep inline styles on top of everything else
        for name in ("bold", "italic", "inlinecode"):
            t.tag_raise(name)

    # -- low level ------------------------------------------------------------
    def _ins(self, text, *tags):
        t = self.text
        stick = t.yview()[1] >= 0.97
        t.configure(state="normal")
        t.insert("end", text, tags)
        t.configure(state="disabled")
        if stick:
            t.see("end")

    def _apply(self, ops):
        t = self.text
        stick = t.yview()[1] >= 0.97
        t.configure(state="normal")
        if self.partial:
            t.delete("pmark", "end-1c")
            self.partial = False
        for text, tags in ops:
            t.insert("end", text, tags)
        p = self.md.partial()
        if p:
            t.mark_set("pmark", "end-1c")
            t.mark_gravity("pmark", "left")
            t.insert("end", p[0], p[1])
            self.partial = True
        t.configure(state="disabled")
        if stick:
            t.see("end")

    # -- public ---------------------------------------------------------------
    def clear(self):
        self.end_stream()
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def welcome(self, model, prompts, on_pick):
        self._ins("\n◆  %s\n" % APP, "welcome_title")
        self._ins("Local coding agent  ·  %s\nAsk me to build, fix, run or research something.\n" % model, "welcome_sub")
        for i, (label, prompt) in enumerate(prompts):
            tag = "chip%d" % i
            self.text.tag_configure(tag, font=self.f["ui_s"], foreground=T["accent"], background=T["card2"],
                                    justify="center")
            self.text.tag_bind(tag, "<Button-1>", lambda e, p=prompt: on_pick(p))
            self.text.tag_bind(tag, "<Enter>", lambda e: self.text.configure(cursor="hand2"))
            self.text.tag_bind(tag, "<Leave>", lambda e: self.text.configure(cursor="arrow"))
            self._ins("  %s  " % label, "chip_row", tag)
            self._ins("  ", "chip_row")
        self._ins("\n", "chip_row")

    def user(self, text):
        self.end_stream()
        self._ins("\nYOU\n", "user_label")
        self._ins(text + "\n", "user")

    def stream(self, kind, chunk, show_thinking=True):
        if kind != self.kind:
            self.end_stream()
            self.kind = kind
            if kind == "thinking":
                if show_thinking:
                    self._ins("\n◌  THINKING\n", "think_label")
            else:
                self._ins("\n●  CODER\n", "asst_label")
        if kind == "thinking":
            if show_thinking:
                self._ins(chunk, "think")
            return
        self._apply(self.md.feed(chunk))

    def end_stream(self):
        if self.kind == "content":
            self._apply(self.md.flush())
        elif self.kind == "thinking":
            self._ins("\n", "think")
        self.partial = False
        self.kind = None

    def tool_call(self, head, body=""):
        self.end_stream()
        self._ins("\n⚙  %s\n" % head, "tool_head")
        if body:
            self._ins(ui_clip(body, 18, 1800) + "\n", "tool_body")

    def tool_result(self, text):
        self.end_stream()
        bad = text.startswith(("ERROR", "WEB SEARCH ERROR", "NO RESULTS")) or "exit_code: " in text[:14] and not text.startswith("exit_code: 0")
        self._ins(("✗  " if bad else "✓  ") + ui_clip(text).replace("\n", "\n    ") + "\n",
                  "result_err" if bad else "result_ok")

    def info(self, text):
        self.end_stream()
        self._ins(text + "\n", "info")

    def error(self, text):
        self.end_stream()
        self._ins("\n✗  " + text + "\n", "error")


# ----------------------------------------------------------------------------
# Dialog helpers
# ----------------------------------------------------------------------------
def center_on(win, parent, w, h):
    win.update_idletasks()
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 3)
    win.geometry("%dx%d+%d+%d" % (w, h, x, y))
    dark_titlebar(win)


def ask_approval(root, fonts, kind, text):
    """Modal approval card. Returns 'once', 'session' or 'deny'."""
    d = tk.Toplevel(root)
    d.title("Approval needed")
    d.configure(bg=T["bg"])
    d.transient(root)
    d.resizable(False, False)
    result = {"v": "deny"}
    title = {"command": "Run this command?", "write": "Write to your files?"}.get(kind, "Allow this action?")
    tk.Label(d, text="⚠  " + title, font=fonts["h3"], bg=T["bg"], fg=T["warn"]).pack(anchor="w", padx=20, pady=(18, 2))
    tk.Label(d, text="The agent is asking for permission:", font=fonts["ui_s"], bg=T["bg"], fg=T["dim"]).pack(anchor="w", padx=20)
    box = tk.Frame(d, bg=T["border"], padx=1, pady=1)
    box.pack(fill="x", padx=20, pady=12)
    txt = tk.Text(box, height=min(10, max(3, text.count("\n") + 2)), width=70, wrap="word", font=fonts["mono"],
                  bg=T["code_bg"], fg=T["code_fg"], relief="flat", bd=0, padx=10, pady=8, highlightthickness=0)
    txt.insert("1.0", text)
    txt.configure(state="disabled")
    txt.pack(fill="x")
    row = tk.Frame(d, bg=T["bg"])
    row.pack(fill="x", padx=20, pady=(0, 18))

    def done(v):
        result["v"] = v
        d.destroy()

    FlatButton(row, "Deny", lambda: done("deny"), "soft", fonts["ui"]).pack(side="right", padx=(8, 0))
    FlatButton(row, "Allow once", lambda: done("once"), "primary", fonts["ui_b"]).pack(side="right", padx=(8, 0))
    FlatButton(row, "Allow all %ss this session" % kind, lambda: done("session"), "ghost", fonts["ui_s"]).pack(side="left")
    d.bind("<Return>", lambda e: done("once"))
    d.bind("<Escape>", lambda e: done("deny"))
    d.protocol("WM_DELETE_WINDOW", lambda: done("deny"))
    d.update_idletasks()
    center_on(d, root, d.winfo_reqwidth(), d.winfo_reqheight())
    try:
        d.grab_set()
    except tk.TclError:
        pass
    d.focus_force()
    root.wait_window(d)
    return result["v"]


def dark_titlebar(win):
    """Dark window title bar on Windows 10/11 (silently ignored elsewhere)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        for attr in (20, 19):
            v = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


WELCOME_PROMPTS = [
    ("Build a snake game", "Create a snake game in Python with tkinter, check that it starts without errors, "
                           "and tell me how to run it."),
    ("Explain this workspace", "Look through the workspace and give me a short overview: structure, main files "
                               "and what the project does."),
    ("Research something", "Find out what is new in the latest stable Python release and summarise the key "
                           "changes with sources."),
    ("Write & test C code", "Write a C program that prints the first 20 Fibonacci numbers, compile it and run it."),
]


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.messages = []
        self.worker = None
        self.session_allowed = set()
        self.tree_paths = {}
        self.state = "idle"
        self.status_text = "Ready"
        self.spin = 0
        self.tools = Toolbox(self.cfg, self.approve)

        root.title(APP)
        root.geometry("1240x800")
        root.minsize(900, 560)
        root.configure(bg=T["bg"])

        ui = pick_font(root, ["Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Inter", "Helvetica Neue",
                              "Ubuntu", "DejaVu Sans"], "TkDefaultFont")
        mono = pick_font(root, ["Cascadia Code", "Cascadia Mono", "JetBrains Mono", "Consolas", "SF Mono",
                                "Menlo", "DejaVu Sans Mono", "Courier New"], "Courier")
        F = tkfont.Font
        self.f = {
            "ui_family": ui, "mono_family": mono,
            "ui": F(family=ui, size=11), "ui_b": F(family=ui, size=11, weight="bold"),
            "ui_i": F(family=ui, size=11, slant="italic"), "ui_s": F(family=ui, size=9),
            "label": F(family=ui, size=8, weight="bold"),
            "h1": F(family=ui, size=19, weight="bold"), "h2": F(family=ui, size=14, weight="bold"),
            "h3": F(family=ui, size=12, weight="bold"),
            "mono": F(family=mono, size=10), "mono_b": F(family=mono, size=10, weight="bold"),
            "mono_s": F(family=mono, size=9),
        }
        root.option_add("*TCombobox*Listbox.background", T["card"])
        root.option_add("*TCombobox*Listbox.foreground", T["text"])
        root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
        root.option_add("*TCombobox*Listbox.selectForeground", T["accent_fg"])
        root.option_add("*TCombobox*Listbox.font", self.f["ui"])

        self.v_auto = tk.BooleanVar(value=self.cfg["auto_approve"])
        self.v_think = tk.BooleanVar(value=self.cfg["think"])
        self.v_show = tk.BooleanVar(value=self.cfg["show_thinking"])
        self.v_writes = tk.BooleanVar(value=self.cfg["confirm_writes"])
        self.v_web = tk.StringVar(value=self.cfg["web_mode"])
        self.v_model = tk.StringVar(value=self.cfg["model"])

        self.style_ttk()
        self.build_ui()
        self.build_menu()
        self.refresh_tree()
        self.show_welcome()
        self.fetch_models()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(40, self.poll)
        root.after(100, self.tick)
        root.after(60, lambda: dark_titlebar(root))

    # ---- styling ------------------------------------------------------------
    def style_ttk(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Dark.TCombobox", fieldbackground=T["input"], background=T["card2"], foreground=T["text"],
                    bordercolor=T["border"], lightcolor=T["border"], darkcolor=T["border"], arrowcolor=T["dim"],
                    padding=6, selectbackground=T["input"], selectforeground=T["text"])
        s.map("Dark.TCombobox", fieldbackground=[("readonly", T["input"])], foreground=[("readonly", T["text"])],
              bordercolor=[("focus", T["accent"])])
        s.configure("Dark.TEntry", fieldbackground=T["input"], foreground=T["text"], insertcolor=T["text"],
                    bordercolor=T["border"], lightcolor=T["border"], darkcolor=T["border"], padding=6)
        s.map("Dark.TEntry", bordercolor=[("focus", T["accent"])])
        s.configure("Side.Treeview", background=T["panel"], fieldbackground=T["panel"], foreground=T["text"],
                    borderwidth=0, rowheight=26, font=self.f["ui"])
        s.map("Side.Treeview", background=[("selected", T["sel"])], foreground=[("selected", T["text"])])
        s.layout("Side.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        for orient in ("Vertical", "Horizontal"):
            name = "Dark.%s.TScrollbar" % orient
            s.configure(name, troughcolor=T["bg"], background=T["border"], bordercolor=T["bg"],
                        lightcolor=T["border"], darkcolor=T["border"], arrowcolor=T["dim"], relief="flat", width=10)
            s.map(name, background=[("active", T["faint"]), ("pressed", T["faint"])])
        s.layout("Dark.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {
            "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})], "sticky": "ns"})])

    def menu(self):
        return tk.Menu(self.root, tearoff=0, bg=T["card"], fg=T["text"], activebackground=T["accent"],
                       activeforeground=T["accent_fg"], bd=0, relief="flat", font=self.f["ui_s"])

    # ---- layout -------------------------------------------------------------
    def build_ui(self):
        r, f = self.root, self.f

        top = tk.Frame(r, bg=T["panel"])
        top.pack(fill="x")
        bar = tk.Frame(top, bg=T["panel"])
        bar.pack(fill="x", padx=14, pady=10)
        tk.Label(bar, text="◆", font=f["h2"], bg=T["panel"], fg=T["accent"]).pack(side="left")
        tk.Label(bar, text=APP, font=f["h3"], bg=T["panel"], fg=T["text"]).pack(side="left", padx=(6, 20))
        self.cb_model = ttk.Combobox(bar, textvariable=self.v_model, width=30, style="Dark.TCombobox", font=f["ui"])
        self.cb_model.pack(side="left")
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self.apply_flags())
        self.cb_model.bind("<Return>", lambda e: self.apply_flags())
        FlatButton(bar, "↻", self.fetch_models, font=f["ui"], padx=9).pack(side="left", padx=4)

        self.btn_menu = FlatButton(bar, "☰", self.show_menu, font=f["h3"], padx=10, pady=2)
        self.btn_menu.pack(side="right")
        FlatButton(bar, "⚙", self.settings_dialog, font=f["h3"], padx=10, pady=2).pack(side="right", padx=2)
        FlatButton(bar, "Links", self.manage_links, font=f["ui_s"], padx=10).pack(side="right", padx=2)
        Toggle(bar, "Auto-approve", self.v_auto, self.on_auto_toggle, f["ui_s"]).pack(side="right", padx=(8, 4))
        Toggle(bar, "Show thoughts", self.v_show, self.apply_flags, f["ui_s"]).pack(side="right", padx=4)
        Toggle(bar, "Thinking", self.v_think, self.apply_flags, f["ui_s"]).pack(side="right", padx=4)
        tk.Frame(r, bg=T["border"], height=1).pack(fill="x")

        # status bar (packed before the body so it stays at the bottom)
        sb = tk.Frame(r, bg=T["panel"])
        sb.pack(side="bottom", fill="x")
        tk.Frame(sb, bg=T["border"], height=1).pack(fill="x")
        inner = tk.Frame(sb, bg=T["panel"])
        inner.pack(fill="x", padx=14, pady=5)
        self.dot = tk.Label(inner, text="●", font=f["ui_s"], bg=T["panel"], fg=T["good"])
        self.dot.pack(side="left")
        self.lbl_status = tk.Label(inner, text="Ready", font=f["ui_s"], bg=T["panel"], fg=T["dim"])
        self.lbl_status.pack(side="left", padx=6)
        self.lbl_info = tk.Label(inner, text="", font=f["ui_s"], bg=T["panel"], fg=T["faint"])
        self.lbl_info.pack(side="right")

        pw = tk.PanedWindow(r, orient="horizontal", bg=T["bg"], sashwidth=4, sashrelief="flat", bd=0,
                            opaqueresize=True)
        pw.pack(fill="both", expand=True)

        # sidebar
        left = tk.Frame(pw, bg=T["panel"])
        hdr = tk.Frame(left, bg=T["panel"])
        hdr.pack(fill="x", padx=12, pady=(14, 2))
        tk.Label(hdr, text="WORKSPACE", font=f["label"], bg=T["panel"], fg=T["faint"]).pack(side="left")
        FlatButton(hdr, "Open", self.open_workspace, font=f["ui_s"], padx=6, pady=0).pack(side="right")
        FlatButton(hdr, "↻", self.refresh_tree, font=f["ui_s"], padx=6, pady=0).pack(side="right")
        self.lbl_ws = tk.Label(left, text="", font=f["ui_s"], bg=T["panel"], fg=T["dim"], anchor="w")
        self.lbl_ws.pack(fill="x", padx=12, pady=(0, 8))
        tf = tk.Frame(left, bg=T["panel"])
        tf.pack(fill="both", expand=True, padx=(6, 0))
        self.tree = ttk.Treeview(tf, show="tree", style="Side.Treeview", selectmode="browse")
        tsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview, style="Dark.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.tag_configure("dir", foreground=T["accent"])
        self.tree.tag_configure("file", foreground=T["text"])
        self.tree.bind("<Double-1>", self.open_selected)
        self.tree.bind("<Button-3>", self.tree_menu)
        pw.add(left, minsize=170, width=250)

        # main column
        main = tk.Frame(pw, bg=T["bg"])
        self.chat = ChatView(main, f)
        self.chat.pack(fill="both", expand=True)

        comp = tk.Frame(main, bg=T["bg"])
        comp.pack(fill="x", padx=22, pady=(4, 14))
        self.box = tk.Frame(comp, bg=T["border"], padx=1, pady=1)
        self.box.pack(fill="x")
        card = tk.Frame(self.box, bg=T["input"])
        card.pack(fill="x")
        self.inp = tk.Text(card, height=2, wrap="word", bg=T["input"], fg=T["text"], insertbackground=T["text"],
                           relief="flat", bd=0, padx=14, pady=10, font=f["ui"], highlightthickness=0,
                           selectbackground=T["sel"], undo=True)
        self.inp.pack(fill="x")
        self.inp.bind("<Return>", self.on_enter)
        self.inp.bind("<KeyRelease>", lambda e: self.grow_input())
        self.inp.bind("<FocusIn>", lambda e: self.box.configure(bg=T["accent"]))
        self.inp.bind("<FocusOut>", lambda e: self.box.configure(bg=T["border"]))
        row = tk.Frame(card, bg=T["input"])
        row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(row, text="Web", font=f["ui_s"], bg=T["input"], fg=T["faint"]).pack(side="left", padx=(6, 6))
        Segmented(row, [("approved", "Approved"), ("open", "Open"), ("off", "Off")], self.v_web,
                  self.apply_flags, f["ui_s"]).pack(side="left")
        tk.Label(row, text="Enter to send  ·  Shift+Enter for a new line", font=f["ui_s"], bg=T["input"],
                 fg=T["faint"]).pack(side="left", padx=14)
        self.btn_send = FlatButton(row, "Send  ➤", self.send, "primary", f["ui_b"], padx=16, pady=5)
        self.btn_send.pack(side="right")
        self.btn_stop = FlatButton(row, "■  Stop", self.stop_run, "soft", f["ui"], padx=12, pady=5)
        self.btn_stop.pack(side="right", padx=8)
        self.btn_stop.set_enabled(False)
        pw.add(main, minsize=520, stretch="always")

        self.update_ws_label()
        self.inp.focus_set()

    def build_menu(self):
        m = self.menu()
        m.add_command(label="New chat", command=self.new_chat)
        m.add_command(label="Set workspace folder…", command=self.pick_workspace)
        m.add_command(label="Open workspace folder", command=self.open_workspace)
        m.add_command(label="Save transcript…", command=self.save_transcript)
        m.add_separator()
        m.add_command(label="Manage approved links…", command=self.manage_links)
        m.add_checkbutton(label="Confirm file writes", variable=self.v_writes, command=self.apply_flags)
        m.add_separator()
        m.add_command(label="Check toolchain", command=self.check_toolchain)
        m.add_command(label="Test connection", command=self.fetch_models)
        m.add_separator()
        m.add_command(label="Quit", command=self.on_close)
        self.main_menu = m

    def show_menu(self):
        b = self.btn_menu
        self.main_menu.tk_popup(b.winfo_rootx() - 150, b.winfo_rooty() + b.winfo_height() + 4)

    def show_welcome(self):
        self.chat.welcome(self.cfg["model"], WELCOME_PROMPTS, self.prefill)

    # ---- small helpers ------------------------------------------------------
    def set_status(self, text, state="idle"):
        self.status_text, self.state = text, state
        self.dot.configure(fg={"idle": T["good"], "busy": T["warn"], "error": T["bad"]}.get(state, T["dim"]))
        self.lbl_status.configure(text=text)

    def tick(self):
        if self.state == "busy":
            self.spin = (self.spin + 1) % len(SPINNER)
            self.lbl_status.configure(text="%s  %s" % (SPINNER[self.spin], self.status_text))
        self.root.after(90, self.tick)

    def set_busy(self, busy):
        self.btn_send.set_enabled(not busy)
        self.btn_stop.set_enabled(busy)

    def on_enter(self, e):
        if e.state & 0x0001:            # Shift+Enter -> newline
            return None
        self.send()
        return "break"

    def grow_input(self):
        try:
            lines = self.inp.count("1.0", "end", "displaylines")[0]
        except Exception:
            lines = int(self.inp.index("end-1c").split(".")[0])
        self.inp.configure(height=max(2, min(9, lines)))

    def prefill(self, text):
        self.inp.delete("1.0", "end")
        self.inp.insert("1.0", text)
        self.inp.focus_set()
        self.grow_input()

    def on_auto_toggle(self):
        if self.v_auto.get() and not messagebox.askyesno(
                APP, "Auto-approve lets the AI run commands and scripts without asking you first.\n\nTurn it on?",
                parent=self.root):
            self.v_auto.set(False)
        self.apply_flags()

    def apply_flags(self):
        c = self.cfg
        c["auto_approve"] = self.v_auto.get()
        c["think"] = self.v_think.get()
        c["show_thinking"] = self.v_show.get()
        c["confirm_writes"] = self.v_writes.get()
        c["web_mode"] = self.v_web.get()
        c["model"] = self.v_model.get().strip() or c["model"]
        save_config(c)

    def update_ws_label(self):
        ws = self.cfg["workspace"]
        self.lbl_ws.configure(text=("…" + ws[-34:]) if len(ws) > 36 else ws)

    # ---- workspace / files --------------------------------------------------
    def pick_workspace(self):
        d = filedialog.askdirectory(initialdir=self.cfg["workspace"], title="Workspace folder")
        if d:
            self.cfg["workspace"] = d
            save_config(self.cfg)
            self.update_ws_label()
            self.refresh_tree()

    def open_workspace(self):
        p = str(self.tools.root())
        try:
            if os.name == "nt":
                os.startfile(p)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            messagebox.showerror(APP, str(e))

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.tree_paths = {}
        root = self.tools.root()

        def add(parent, folder, depth):
            try:
                items = sorted(folder.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except OSError:
                return
            for it in items[:200]:
                if it.name in SKIP_DIRS:
                    continue
                iid = self.tree.insert(parent, "end", text=("  " + it.name) if it.is_file() else ("  " + it.name + "/"),
                                       open=depth < 1, tags=("file" if it.is_file() else "dir",))
                self.tree_paths[iid] = it
                if it.is_dir() and depth < 4:
                    add(iid, it, depth + 1)

        add("", root, 0)

    def selected_path(self):
        sel = self.tree.selection()
        return self.tree_paths.get(sel[0]) if sel else None

    def open_selected(self, _=None):
        p = self.selected_path()
        if not p or not p.is_file():
            return
        w = tk.Toplevel(self.root)
        w.title(p.name)
        w.configure(bg=T["bg"])
        center_on(w, self.root, 880, 640)
        head = tk.Frame(w, bg=T["panel"])
        head.pack(fill="x")
        tk.Label(head, text=str(p), font=self.f["ui_s"], bg=T["panel"], fg=T["dim"]).pack(side="left", padx=14, pady=8)
        FlatButton(head, "Copy path", lambda: (w.clipboard_clear(), w.clipboard_append(str(p))), font=self.f["ui_s"],
                   padx=8, pady=2).pack(side="right", padx=10)
        body = tk.Frame(w, bg=T["bg"])
        body.pack(fill="both", expand=True)
        content = p.read_text(encoding="utf-8", errors="replace")[:300000]
        n = content.count("\n") + 1
        gutter = tk.Text(body, width=max(4, len(str(n)) + 1), bg=T["panel"], fg=T["faint"], font=self.f["mono"],
                         relief="flat", bd=0, padx=8, pady=8, highlightthickness=0, state="normal", cursor="arrow")
        gutter.tag_configure("r", justify="right")
        gutter.insert("1.0", "\n".join(str(i) for i in range(1, n + 1)), "r")
        gutter.configure(state="disabled")
        gutter.pack(side="left", fill="y")
        txt = tk.Text(body, wrap="none", bg=T["bg"], fg=T["text"], font=self.f["mono"], relief="flat", bd=0,
                      padx=10, pady=8, highlightthickness=0, insertbackground=T["text"], selectbackground=T["sel"])
        sb = ttk.Scrollbar(body, orient="vertical", style="Dark.Vertical.TScrollbar")

        def on_scroll(lo, hi):
            sb.set(lo, hi)
            gutter.yview_moveto(lo)

        txt.configure(yscrollcommand=on_scroll)
        sb.configure(command=txt.yview)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", content)
        txt.configure(state="disabled")

    def tree_menu(self, ev):
        iid = self.tree.identify_row(ev.y)
        if not iid:
            return
        self.tree.selection_set(iid)
        p = self.selected_path()
        if not p:
            return
        rel = str(p.relative_to(self.tools.root())).replace("\\", "/")
        m = self.menu()
        if p.is_file():
            m.add_command(label="Open", command=self.open_selected)
            m.add_command(label="Ask AI to analyse", command=lambda: self.prefill("Analyse the file %s" % rel))
            m.add_command(label="Ask AI to test it", command=lambda: self.prefill("Run %s and fix any errors." % rel))
        m.add_command(label="Copy relative path",
                      command=lambda: (self.root.clipboard_clear(), self.root.clipboard_append(rel)))
        m.tk_popup(ev.x_root, ev.y_root)

    # ---- dialogs ------------------------------------------------------------
    def dialog(self, title, w, h):
        d = tk.Toplevel(self.root)
        d.title(title)
        d.configure(bg=T["bg"])
        d.transient(self.root)
        center_on(d, self.root, w, h)
        try:
            d.grab_set()
        except tk.TclError:
            pass
        return d

    def settings_dialog(self):
        f = self.f
        d = self.dialog("Settings", 620, 640)
        tk.Label(d, text="Settings", font=f["h2"], bg=T["bg"], fg=T["text"]).pack(anchor="w", padx=24, pady=(20, 10))
        grid = tk.Frame(d, bg=T["bg"])
        grid.pack(fill="x", padx=24)
        fields = [("Ollama host", "host"), ("Default model", "model"), ("Workspace folder", "workspace"),
                  ("Temperature", "temperature"), ("Context size (num_ctx)", "num_ctx"),
                  ("Keep model loaded", "keep_alive"), ("Command timeout (s)", "timeout"),
                  ("Max agent steps", "max_steps")]
        vars_ = {}
        for i, (label, key) in enumerate(fields):
            tk.Label(grid, text=label, font=f["ui_s"], bg=T["bg"], fg=T["dim"]).grid(row=i, column=0, sticky="w", pady=5)
            v = tk.StringVar(value=str(self.cfg.get(key, "")))
            vars_[key] = v
            ttk.Entry(grid, textvariable=v, width=44, style="Dark.TEntry", font=f["ui"]).grid(
                row=i, column=1, sticky="ew", padx=(16, 0), pady=5)
        grid.columnconfigure(1, weight=1)
        tk.Label(d, text="Custom instructions (added to the AI's system prompt)", font=f["ui_s"], bg=T["bg"],
                 fg=T["dim"]).pack(anchor="w", padx=24, pady=(14, 4))
        box = tk.Frame(d, bg=T["border"], padx=1, pady=1)
        box.pack(fill="both", expand=True, padx=24)
        extra = tk.Text(box, height=6, wrap="word", bg=T["input"], fg=T["text"], insertbackground=T["text"],
                        relief="flat", bd=0, padx=10, pady=8, font=f["ui"], highlightthickness=0)
        extra.pack(fill="both", expand=True)
        extra.insert("1.0", self.cfg.get("custom_instructions", ""))

        def ok():
            new = {}
            for key, v in vars_.items():
                val = v.get().strip()
                try:
                    if key == "temperature":
                        new[key] = float(val)
                    elif key in ("num_ctx", "timeout", "max_steps"):
                        new[key] = int(val)
                    else:
                        new[key] = val
                except ValueError:
                    messagebox.showerror(APP, "Invalid value for %s" % key, parent=d)
                    return
            new["custom_instructions"] = extra.get("1.0", "end").strip()
            self.cfg.update(new)
            self.v_model.set(self.cfg["model"])
            save_config(self.cfg)
            self.update_ws_label()
            self.refresh_tree()
            d.destroy()
            self.fetch_models()

        row = tk.Frame(d, bg=T["bg"])
        row.pack(fill="x", padx=24, pady=16)
        FlatButton(row, "Save", ok, "primary", f["ui_b"], padx=22).pack(side="right")
        FlatButton(row, "Cancel", d.destroy, "soft", f["ui"], padx=16).pack(side="right", padx=8)

    def manage_links(self):
        f = self.f
        d = self.dialog("Approved links", 600, 520)
        tk.Label(d, text="Approved sites", font=f["h2"], bg=T["bg"], fg=T["text"]).pack(anchor="w", padx=24, pady=(20, 2))
        tk.Label(d, text="The AI can search and read these sites when Web is set to “Approved”. "
                         "A path (like /3/) limits it to that section.", font=f["ui_s"], bg=T["bg"], fg=T["dim"],
                 wraplength=540, justify="left").pack(anchor="w", padx=24, pady=(0, 10))
        box = tk.Frame(d, bg=T["border"], padx=1, pady=1)
        box.pack(fill="both", expand=True, padx=24)
        lb = tk.Listbox(box, selectmode="extended", bg=T["input"], fg=T["text"], font=f["ui"], relief="flat", bd=0,
                        highlightthickness=0, selectbackground=T["accent"], selectforeground=T["accent_fg"],
                        activestyle="none")
        lb.pack(fill="both", expand=True)
        for l in self.cfg["links"]:
            lb.insert("end", l)
        row = tk.Frame(d, bg=T["bg"])
        row.pack(fill="x", padx=24, pady=14)
        v = tk.StringVar()
        e = ttk.Entry(row, textvariable=v, style="Dark.TEntry", font=f["ui"])
        e.pack(side="left", fill="x", expand=True)

        def commit():
            self.cfg["links"] = list(lb.get(0, "end"))
            save_config(self.cfg)

        def add(_=None):
            s = v.get().strip()
            if not s:
                return
            if "//" not in s:
                s = "https://" + s
            if s not in lb.get(0, "end"):
                lb.insert("end", s)
                commit()
            v.set("")

        def rem():
            for i in reversed(lb.curselection()):
                lb.delete(i)
            commit()

        def reset():
            lb.delete(0, "end")
            for l in DEFAULT_LINKS:
                lb.insert("end", l)
            commit()

        e.bind("<Return>", add)
        FlatButton(row, "Add", add, "primary", f["ui_b"], padx=14).pack(side="left", padx=(8, 0))
        FlatButton(row, "Remove", rem, "soft", f["ui"], padx=12).pack(side="left", padx=6)
        FlatButton(row, "Reset", reset, "ghost", f["ui_s"], padx=10).pack(side="left")
        e.focus_set()

    def check_toolchain(self):
        names = ["powershell", "pwsh", "cmd", "bash", "python", "py", "gcc", "g++", "make", "lua", "node", "git"]
        lines = ["python: %s" % sys.executable, "platform: %s" % platform.platform(), ""]
        for n in names:
            lines.append("%-11s %s" % (n, shutil.which(n) or "- not found"))
        messagebox.showinfo("Toolchain", "\n".join(lines), parent=self.root)

    def save_transcript(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            Path(p).write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")

    def new_chat(self):
        if self.worker and self.worker.is_alive():
            return
        self.messages = []
        self.tools.cache.clear()
        self.chat.clear()
        self.show_welcome()
        self.lbl_info.configure(text="")

    # ---- models -------------------------------------------------------------
    def fetch_models(self):
        host = self.cfg["host"].rstrip("/")
        self.set_status("Connecting to Ollama…", "busy")

        def work():
            try:
                with urllib.request.urlopen(host + "/api/tags", timeout=5) as r:
                    names = [m["name"] for m in json.loads(r.read())["models"]]
                self.q.put(("models", names))
                self.q.put(("status", "Connected to Ollama · %d model%s" % (len(names), "" if len(names) == 1 else "s"), "idle"))
            except Exception as e:
                self.q.put(("status", "Cannot reach Ollama at %s (%s)" % (host, e), "error"))

        threading.Thread(target=work, daemon=True).start()

    # ---- approvals (called from the worker thread) ---------------------------
    def approve(self, kind, text):
        if self.cfg["auto_approve"] or kind in self.session_allowed:
            return True
        ev, box = threading.Event(), {}
        self.q.put(("approve", kind, text, ev, box))
        while not ev.wait(0.2):
            if self.stop.is_set():
                return False
        return box.get("ok", False)

    # ---- the AI's instructions ----------------------------------------------
    def system_prompt(self):
        n, c = now_info(), self.cfg
        mode = c["web_mode"]
        hosts = self.tools._approved_hosts()
        if mode == "off":
            web = ("DISABLED by the user. You have no web tools. Answer from your own knowledge and say clearly "
                   "when something may be out of date.")
        elif mode == "open":
            web = "OPEN - any public website can be read. Prefer official documentation and primary sources."
        else:
            shown = [re.sub(r"^https?://(www\.)?", "", l).rstrip("/") for l in c["links"]]
            web = "APPROVED SITES ONLY - " + ", ".join(shown) + ". Other sites are blocked."

        L = [
            "You are %s, an expert software engineer working autonomously in a workspace folder on the user's own "
            "computer. You can inspect, create, edit, run, compile and test code, and research documentation on the "
            "web. You act through tools, then report the results honestly." % APP,
            "",
            "## Right now",
            "- Current date and time: %s (timezone: %s)  |  ISO %s" % (n["pretty"], n["tz"], n["iso"]),
            "- Treat this as ground truth for \"today\", \"now\", \"this year\" and \"latest\". Your training data is "
            "older than this date, so versions, APIs, prices and news may have changed: verify time-sensitive facts "
            "with the web tools and include the current year (%d) in searches for latest/current information. "
            "Call get_datetime if you need the exact time again later in a long session." % n["year"],
            "- System: %s; Python %s. Workspace: %s" % (platform.platform(), platform.python_version(), self.tools.root()),
            "  All tool paths are relative to the workspace and commands run with it as the working directory.",
            "- Web access: " + web,
            "",
            "## How to work",
            "1. Orient first. For existing code use list_files, search_files and read_file before changing anything. "
            "Never guess at the contents of a file you have not read.",
            "2. For anything with several steps, state a short plan (2-4 lines), then start executing. Do not ask "
            "permission for obvious steps. Ask a question only when a missing detail truly blocks you, and then ask "
            "just that one question.",
            "3. Write complete, working code. write_file needs the ENTIRE file - never placeholders like "
            "\"... rest of code ...\". Prefer edit_file for small changes; read the file first and copy `old` exactly.",
            "4. Verify every change by running it (run_python, compile_c, run_command). If it fails, read the error, "
            "fix the root cause and run again. After 3 failed attempts at the same problem, change approach or "
            "explain what is blocking you. Programs cannot read stdin here: avoid input(), test with arguments or "
            "constants.",
            "5. Never claim something works unless a tool result showed it. Quote the key output when reporting.",
            "6. Finish with a short summary: files created or changed, how to run them, and caveats. Do not paste "
            "whole files you already wrote.",
            "",
            "## Tool rules",
            "- Call tools only through the function-calling interface. Never print a tool call as text or JSON.",
            "- Use exact argument names from the tool schemas. Do not repeat an identical call hoping for a "
            "different result.",
            "- Paths use forward slashes and must stay inside the workspace.",
            "- C/C++: always build with compile_c (it picks gcc/g++, returns errors, run_after=true runs it). Do not "
            "call gcc through run_command.",
            "- Keep command output small: filter or limit it.",
        ]
        if os.name == "nt":
            L.append("- This is Windows: run_command uses PowerShell by default. Write PowerShell syntax "
                     "(not bash), e.g. Get-ChildItem instead of ls -la.")
        if mode != "off":
            L += [
                "",
                "## Research protocol (web tools)",
                "Research when: you use a library/API/flag/version you are not 100% sure about; you debug an "
                "unfamiliar error; the user asks about latest/current/news/releases/prices or anything newer than "
                "your training data; or the user asks you to look something up. Do NOT research basic syntax you "
                "already know well.",
                "1. Start with web_research(query). One call searches, opens the best pages and returns only the "
                "passages relevant to your query. That is usually all you need.",
                "2. Write queries as KEYWORDS, not sentences: 3-8 words with the exact identifier, library, language "
                "and version/year. Good: \"pathlib Path.rglob python 3.12\", \"SDL_CreateWindow flags SDL2\". Bad: "
                "\"how can I search for files in folders with python\". For errors, use the distinctive part of the "
                "message.",
                "3. Weak or empty results? Do not give up and do not repeat the same query. Rephrase with different "
                "keywords (official names, synonyms, drop the version), aim at one site, or open the official page "
                "directly with fetch_url using a URL pattern below.",
                "4. Need a whole page? fetch_url(url). Long page? fetch_url(url, find=\"keywords\") jumps to the "
                "matching parts, page=2 continues. web_search only LISTS results - follow up with fetch_url.",
                "5. Read before you answer. Base your answer on what the pages actually say. After using the web, "
                "end with \"Sources:\" and the URLs you used. If you could not find it, say so plainly instead of "
                "guessing.",
                "6. Budget: 1-3 research calls per question; stop as soon as you have enough.",
            ]
            patterns = [h for k, h in DOC_HINTS.items() if mode == "open" or any(k in x for x in hosts)]
            if patterns:
                L.append("Direct URL patterns for fetch_url:")
                L += ["  - " + p for p in patterns]
        L += [
            "",
            "## Style",
            "Be concise and direct, no filler. Use markdown; put code in fenced blocks with a language tag. Reply in "
            "the language the user writes in.",
        ]
        if c.get("custom_instructions", "").strip():
            L += ["", "## Extra instructions from the user", c["custom_instructions"].strip()]
        return "\n".join(L)

    # ---- chat loop ----------------------------------------------------------
    def send(self):
        if self.worker and self.worker.is_alive():
            return
        text = self.inp.get("1.0", "end").strip()
        if not text:
            return
        self.apply_flags()
        self.inp.delete("1.0", "end")
        self.grow_input()
        if not self.messages:
            self.chat.clear()
        self.messages.append({"role": "user", "content": text})
        self.chat.user(text)
        self.stop.clear()
        self.set_busy(True)
        self.set_status("Thinking…", "busy")
        self.worker = threading.Thread(target=self.agent, daemon=True)
        self.worker.start()

    def stop_run(self):
        self.stop.set()
        self.tools.kill()
        self.set_status("Stopping…", "busy")

    def fmt_call(self, name, a):
        a = Toolbox.parse_args(a)
        if name == "write_file":
            return "write_file  %s" % a.get("path"), str(a.get("content", ""))
        if name == "edit_file":
            return "edit_file  %s" % a.get("path"), "--- old\n%s\n+++ new\n%s" % (
                str(a.get("old", ""))[:1200], str(a.get("new", ""))[:1200])
        if name == "run_command":
            return "run_command  [%s]" % a.get("shell", "auto"), str(a.get("command", ""))
        if name in ("web_research", "web_search"):
            return "%s  “%s”" % (name, a.get("query", "")), ""
        if name == "fetch_url":
            extra = ("   find: %s" % a["find"]) if a.get("find") else (("   page %s" % a["page"]) if a.get("page") else "")
            return "fetch_url  %s%s" % (a.get("url", ""), extra), ""
        if name == "read_file":
            rng = ("  [%s-%s]" % (a.get("start_line", 1), a.get("end_line", "…"))) if a.get("start_line") or a.get("end_line") else ""
            return "read_file  %s%s" % (a.get("path"), rng), ""
        brief = ", ".join("%s=%s" % (k, str(v)[:60]) for k, v in a.items())
        return "%s  %s" % (name, brief[:140]), ""

    def agent(self):
        seen = {}
        nudged = False
        try:
            for step in range(int(self.cfg["max_steps"])):
                if self.stop.is_set():
                    break
                self.q.put(("status", "Thinking…", "busy"))
                msg = self.stream_chat()
                if msg is None:
                    break
                self.messages.append(msg)
                calls = msg.get("tool_calls") or []
                if not calls:
                    if not (msg.get("content") or "").strip() and step > 0 and not nudged:
                        nudged = True
                        self.messages.append({"role": "user", "content": "Continue: either call the next tool you "
                                              "need, or give your final answer to the user now."})
                        continue
                    break
                for c in calls:
                    if self.stop.is_set():
                        break
                    fn = c.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments") or {}
                    self.q.put(("tool_call", name, args))
                    key = (name, json.dumps(Toolbox.parse_args(args), sort_keys=True, default=str))
                    if name in WEB_TOOLS:
                        seen[key] = seen.get(key, 0) + 1
                    if name in WEB_TOOLS and seen[key] > 1:
                        result = ("NOTICE: you already made this exact call. Do not repeat it. Use different "
                                  "keywords, open a specific page with fetch_url (use find= or page=), or answer "
                                  "with the information you already have.")
                    else:
                        result = self.tools.call(name, args)
                    self.q.put(("tool_result", name, result))
                    tool_msg = {"role": "tool", "tool_name": name, "content": clip(result, 12000)}
                    if c.get("id"):
                        tool_msg["tool_call_id"] = c["id"]
                    self.messages.append(tool_msg)
            else:
                self.q.put(("info", "Reached the step limit. Say “continue” to keep going."))
        except Exception as e:
            self.q.put(("error", str(e)))
        finally:
            self.q.put(("done",))

    def _ollama_messages(self):
        msgs = []
        for m in self.messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "content": m.get("content", ""), "tool_name": m.get("tool_name", "")})
            else:
                entry = {"role": m["role"], "content": m.get("content", "")}
                if m.get("tool_calls"):
                    entry["tool_calls"] = m["tool_calls"]
                msgs.append(entry)
        # keep the conversation inside the context window (~3 chars per token)
        budget = max(int(self.cfg["num_ctx"]) * 3 - 9000, 6000)

        def total():
            return sum(len(m.get("content", "")) + 60 for m in msgs)

        if total() > budget:
            for m in msgs[:-4]:
                if m["role"] == "tool" and len(m["content"]) > 700:
                    m["content"] = m["content"][:500] + "\n...[old tool output shortened to save context]..."
                if total() <= budget:
                    break
        while total() > budget:
            nxt = next((i for i in range(1, len(msgs)) if msgs[i]["role"] == "user"), None)
            if nxt is None:
                break
            del msgs[:nxt]
        return msgs

    def stream_chat(self):
        cfg = self.cfg
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "system", "content": self.system_prompt()}] + self._ollama_messages(),
            "tools": self.tools.schemas(),
            "stream": True,
            "think": bool(cfg["think"]),
            "keep_alive": cfg.get("keep_alive") or "30m",
            "options": {"temperature": float(cfg["temperature"]), "num_ctx": int(cfg["num_ctx"])},
        }
        resp = None
        for attempt in range(3):
            req = urllib.request.Request(cfg["host"].rstrip("/") + "/api/chat",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            try:
                resp = urllib.request.urlopen(req, timeout=900)
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                low = body.lower()
                if attempt < 2 and "think" in low and "think" in payload:
                    payload.pop("think")
                    continue
                if attempt < 2 and "tools" in low and payload.get("tools"):
                    payload["tools"] = []
                    self.q.put(("info", "This model does not support tools - continuing without them."))
                    continue
                raise RuntimeError("Ollama HTTP %s: %s" % (e.code, body[:500]))
            except urllib.error.URLError as e:
                raise RuntimeError("Cannot reach Ollama at %s: %s" % (cfg["host"], e.reason))
        if resp is None:
            raise RuntimeError("No response from Ollama.")

        content, thinking, calls = "", "", []
        with resp:
            for line in resp:
                if self.stop.is_set():
                    return None
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                m = chunk.get("message") or {}
                if m.get("thinking"):
                    thinking += m["thinking"]
                    self.q.put(("thinking", m["thinking"]))
                if m.get("content"):
                    content += m["content"]
                    self.q.put(("content", m["content"]))
                if m.get("tool_calls"):
                    calls.extend(m["tool_calls"])
                if chunk.get("done"):
                    used = (chunk.get("prompt_eval_count") or 0) + (chunk.get("eval_count") or 0)
                    dur = (chunk.get("eval_duration") or 0) / 1e9
                    tps = (chunk.get("eval_count") or 0) / dur if dur else 0
                    self.q.put(("stats", used, tps))
        msg = {"role": "assistant", "content": content}
        if thinking:
            msg["thinking"] = thinking
        if calls:
            msg["tool_calls"] = calls
        return msg

    # ---- UI event pump ------------------------------------------------------
    def poll(self):
        try:
            for _ in range(80):
                item = self.q.get_nowait()
                k = item[0]
                if k in ("thinking", "content"):
                    self.chat.stream(k, item[1], self.cfg["show_thinking"])
                elif k == "tool_call":
                    head, body = self.fmt_call(item[1], item[2])
                    self.chat.tool_call(head, body)
                    self.set_status("Running %s…" % item[1], "busy")
                elif k == "tool_result":
                    self.chat.tool_result(item[2])
                    if item[1] in ("write_file", "edit_file", "run_command", "run_python", "compile_c"):
                        self.refresh_tree()
                    self.set_status("Thinking…", "busy")
                elif k == "info":
                    self.chat.info(item[1])
                elif k == "error":
                    self.chat.error(item[1])
                    self.set_status("Error", "error")
                elif k == "status":
                    running = bool(self.worker and self.worker.is_alive())
                    if not running or item[2] != "idle":
                        self.set_status(item[1], item[2])
                elif k == "models":
                    self.cb_model["values"] = item[1]
                elif k == "stats":
                    pct = 100 * item[1] // max(1, int(self.cfg["num_ctx"]))
                    self.lbl_info.configure(text="context %d%%   ·   %.0f tok/s" % (pct, item[2]))
                elif k == "approve":
                    _, kind, text, ev, box = item
                    res = ask_approval(self.root, self.f, kind, text)
                    if res == "session":
                        self.session_allowed.add(kind)
                    box["ok"] = res != "deny"
                    ev.set()
                elif k == "done":
                    self.chat.end_stream()
                    self.set_busy(False)
                    if self.stop.is_set():
                        self.chat.info("[stopped]")
                    if self.state != "error":
                        self.set_status("Ready", "idle")
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def on_close(self):
        self.stop.set()
        self.tools.kill()
        self.apply_flags()
        self.root.destroy()


def main():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
