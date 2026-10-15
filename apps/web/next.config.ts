import type { NextConfig } from "next";

const API = process.env.PETTI_API_URL ?? "http://localhost:8765";

if (process.env.VERCEL === "1" && !process.env.PETTI_API_URL) {
  console.warn(
    "[petti] PETTI_API_URL тохируулаагүй байна — Vercel дээр /api/* proxy ажиллахгүй. Environment Variables дээр model server-ийн URL оруулна уу.",
  );
}

const nextConfig: NextConfig = {
  output: "standalone",
  // Browser calls /api/* → proxied to the FastAPI model server (no CORS issues)
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/:path*` },
      { source: "/images/:path*", destination: `${API}/images/:path*` },
    ];
  },
  experimental: { proxyTimeout: 120_000 },
};

export default nextConfig;
