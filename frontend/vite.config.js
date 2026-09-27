export default {
  server:{
    proxy:{
      '/api':{
        target:process.env.VITE_API_PROXY_TARGET||'http://localhost:8765',
        changeOrigin:true,
        rewrite:path=>path.replace(/^\/api/,'')
      }
    }
  }
};
