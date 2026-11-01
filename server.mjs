import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

try {process.loadEnvFile(fileURLToPath(new URL("./.env", import.meta.url)));} catch(error) {if(error.code!=="ENOENT") throw error;}

const root = fileURLToPath(new URL(".", import.meta.url));
const port = Number(process.env.PORT || 4173);
const trainerUrl = process.env.TRAINER_URL || "http://127.0.0.1:8000";
const mime = { ".html":"text/html; charset=utf-8", ".css":"text/css; charset=utf-8", ".js":"text/javascript; charset=utf-8", ".json":"application/json; charset=utf-8" };

async function callTrainer(path, request) {
  const response = await fetch(trainerUrl + path, {
    method: request.method,
    headers: { "content-type": request.headers["content-type"] || "application/json" },
    body: ["GET", "HEAD"].includes(request.method) ? undefined : request,
    duplex: ["GET", "HEAD"].includes(request.method) ? undefined : "half"
  });
  return { status: response.status, type: response.headers.get("content-type") || "application/json", body: Buffer.from(await response.arrayBuffer()) };
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url, "http://" + request.headers.host);
  if (url.pathname === "/api/hardware") {
    try {
      const result = await callTrainer("/hardware", request);
      response.writeHead(result.status, { "content-type": result.type }); response.end(result.body);
    } catch {
      response.writeHead(200, {"content-type":"application/json"});
      response.end(JSON.stringify({name:"GPU not detected", memoryGb:0, trainerOnline:false}));
    }
    return;
  }
  if (url.pathname.startsWith("/api/")) {
    try {
      const result = await callTrainer(url.pathname.slice(4) + url.search, request);
      response.writeHead(result.status, { "content-type": result.type }); response.end(result.body);
    } catch {
      response.writeHead(503, {"content-type":"application/json"});
      response.end(JSON.stringify({detail:"The WSL trainer is offline. Start trainer/start.sh, then try again."}));
    }
    return;
  }
  const safePath = normalize(decodeURIComponent(url.pathname)).replace(/^([.][.][\\/])+/, "");
  try {
    const path = join(root, url.pathname === "/" ? "index.html" : safePath);
    const data = await readFile(path);
    response.writeHead(200, {"content-type":mime[extname(path)] || "application/octet-stream"});
    response.end(data);
  } catch {
    const data = await readFile(join(root, "index.html"));
    response.writeHead(200, {"content-type":mime[".html"]});
    response.end(data);
  }
});

server.listen(port, "127.0.0.1", () => console.log("ForgeTune running at http://127.0.0.1:" + port));
