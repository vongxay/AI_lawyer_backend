# AI Lawyer Backend - Quick Start Guide
## Get Running in 5 Minutes

---

## 🚀 **IMMEDIATE FIXES** (Do This First)

### **Step 1: Free Up Port 8000**

Your port 8000 is currently in use by process ID 6896.

**Windows PowerShell (Run as Administrator):**
```powershell
# Kill the process using port 8000
taskkill /PID 6896 /F

# Verify port is free
netstat -ano | findstr :8000
```

If you still see output, repeat with the new PID shown.

---

### **Step 2: Install Dependencies**

```powershell
# Navigate to project directory
cd D:\cms\AI_lawyer_backend

# Upgrade pip
python -m pip install --upgrade pip

# Install ALL dependencies
pip install -r requirements.txt
```

**Note:** The `pyiceberg` warning you saw is NOT a problem - it's an optional dependency that Supabase includes but we don't directly use. You can safely ignore that error.

---

### **Step 3: Set Up Redis** (Choose One Option)

**Option A: Using Docker (Recommended)**
```powershell
docker run -d -p 6379:6379 --name redis-cache redis:7-alpine
```

**Option B: Skip for Now** (App will work with in-memory fallback)

---

### **Step 4: Configure Environment**

```powershell
# Copy example environment file
Copy-Item .env.example .env

# Edit .env file - at minimum set these:
notepad .env
```

**Minimum Required Settings:**
```env
APP_ENV=development

# Security - generate a random string
JWT_SECRET=your-random-32-character-secret-key-here

# AI Providers - at least one required
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional but recommended
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
REDIS_URL=redis://localhost:6379/0
```

---

### **Step 5: Start the Server**

```powershell
# Development mode with auto-reload
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
{"name": "ai-lawyer-backend", "version": "2.0.0", "env": "development", "event": "app.startup", ...}
INFO:     Application startup complete.
```

**Access Points:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

## ✅ **VERIFICATION CHECKLIST**

Test each component:

### **1. Server Running**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "degraded",
  "redis": false,
  "supabase": false,
  "version": "2.0.0"
}
```

> Note: "degraded" is OK for initial testing without Redis/Supabase configured.

### **2. API Documentation**
Open browser: http://localhost:8000/docs

You should see the full Swagger UI with all endpoints.

### **3. Test Legal Query Endpoint**

In Swagger UI or using curl:

```bash
curl -X POST "http://localhost:8000/api/v1/legal/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements for a valid contract under Thai law?", "jurisdiction": "TH"}'
```

Expected: IRAC-structured response (may be stub results without database).

---

## 🗄️ **DATABASE SETUP** (Optional for Full Functionality)

### **Using Supabase (Recommended)**

1. **Create Account:** https://supabase.com

2. **Create Project:**
   - Choose organization
   - Create new project
   - Select region closest to you
   - Wait for setup (~2 minutes)

3. **Get Credentials:**
   - Go to Settings → API
   - Copy `Project URL` → `SUPABASE_URL`
   - Copy `service_role` key → `SUPABASE_KEY`

4. **Run Migration:**
   - Go to SQL Editor in Supabase dashboard
   - Copy entire contents of `database_migration.sql`
   - Paste and click "Run"
   - Wait for success message

5. **Update .env:**
   ```env
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_KEY=eyJhbGc...your-key-here
   ```

6. **Restart Server** and test `/health` again - should show `"supabase": true`

---

## 🔧 **TROUBLESHOOTING**

### **Problem: Port Still in Use**

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill it (replace XXXX with PID)
taskkill /PID XXXX /F

# Alternative: Use different port
python -m uvicorn main:app --reload --port 8001
```

### **Problem: pyiceberg Installation Error**

**This is NORMAL and EXPECTED.** The app works fine without it.

The error occurs because:
- `supabase` package lists `pyiceberg` as an optional dependency
- `pyiceberg` requires Visual C++ Build Tools on Windows
- We don't actually use pyiceberg in our code

**Solution:** Ignore the warning. The app will work perfectly.

### **Problem: Redis Connection Refused**

```powershell
# Check if Redis is running
docker ps | grep redis

# If not running, start it
docker run -d -p 6379:6379 --name redis-cache redis:7-alpine

# Or disable in .env (will use in-memory fallback)
REDIS_URL=
```

### **Problem: Module Not Found Errors**

```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### **Problem: Server Won't Start**

Check logs carefully. Common issues:
- Missing environment variables → Check `.env`
- Port conflict → Free up port 8000
- Invalid JWT secret → Make it at least 32 characters

---

## 📊 **NEXT STEPS AFTER STARTUP**

Once the server is running:

### **1. Explore the API**

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000

# View docs
open http://localhost:8000/docs
```

### **2. Test Basic Functionality**

Try these in Swagger UI (`/docs`):

1. **Legal Query:**
   - Endpoint: `POST /api/v1/legal/query`
   - Question: "What is the legal drinking age in Thailand?"

2. **Document Analysis:**
   - Endpoint: `POST /api/v1/documents/analyze`
   - Upload a PDF contract

3. **Health Check:**
   - Endpoint: `GET /health`
   - Should return status and version

### **3. Monitor Logs**

Watch for these log messages:
- `app.startup` - Server started successfully
- `llm.generate.ok` - LLM calls working
- `retriever.hybrid_search.ok` - RAG pipeline working
- `orchestrator.done` - Full query processed

### **4. Set Up Development Environment**

For ongoing development:

```powershell
# Install dev tools
pip install pytest pytest-cov black ruff

# Run tests
pytest

# Format code
black .

# Lint
ruff check .
```

---

## 🎯 **PRODUCTION DEPLOYMENT**

When ready for production:

### **1. Review Production Checklist**

See `PRODUCTION_READINESS_REPORT.md` for complete checklist.

### **2. Set Up Production Services**

- ✅ Supabase project (production)
- ✅ Redis cluster (production)
- ✅ Sentry account for error tracking
- ✅ Domain and SSL certificate

### **3. Deploy**

**Option A: Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Option B: Cloud Platform**
- Railway.app
- Render.com
- AWS ECS/Fargate
- Google Cloud Run

See `DEPLOYMENT.md` for detailed instructions.

---

## 📞 **SUPPORT & RESOURCES**

### **Documentation Files:**
- `DEPLOYMENT.md` - Complete deployment guide
- `PRODUCTION_READINESS_REPORT.md` - Architecture assessment
- `AI_Lawyer_Architecture.md` - System design
- `.env.example` - Environment variable reference

### **Key Endpoints:**
- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /api/v1/legal/query` - Legal Q&A
- `POST /api/v1/documents/analyze` - Document review
- `POST /api/v1/evidence/analyze` - Evidence analysis

### **Common Commands:**

```powershell
# Start server
python -m uvicorn main:app --reload

# Run tests
pytest --cov=.

# Check code quality
ruff check .
black --check .

# View logs (if using Docker)
docker logs ai-lawyer-backend
```

---

## ✅ **SUCCESS CRITERIA**

You'll know everything is working when:

1. ✅ Server starts without errors
2. ✅ `/health` returns status 200
3. ✅ Can access `/docs` and see all endpoints
4. ✅ Legal query returns IRAC-structured response
5. ✅ No critical errors in logs
6. ✅ Redis connected (if configured)
7. ✅ Supabase connected (if configured)

---

## 🎉 **YOU'RE READY!**

Your AI Lawyer backend is now running and ready for development!

**Next Steps:**
1. Test basic functionality
2. Configure Supabase for full features
3. Set up Redis for rate limiting
4. Review architecture documentation
5. Plan your first legal query workflow

Happy coding! ⚖️🤖
