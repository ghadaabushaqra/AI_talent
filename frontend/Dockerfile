FROM node:24-alpine
WORKDIR /web
COPY package.json pnpm-lock.yaml* ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY . .
EXPOSE 5173
CMD ["pnpm","dev","--host","0.0.0.0","--port","5173"]
