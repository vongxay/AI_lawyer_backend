# AI Lawyer Backend - Production Deployment Guide
## Version 2.0.0 - Production Ready

---

## 📋 **PREREQUISITES**

### **Required Services**
1. **Python 3.11+** (tested on 3.14)
2. **PostgreSQL 15+** with pgvector extension (via Supabase or self-hosted)
3. **Redis 7+** (for caching and rate limiting)
4. **LLM API Keys**: OpenAI + Anthropic (at least one required)

### **Optional but Recommended**
- Docker & Docker Compose
- Doppler or similar for secrets management
- Sentry account for error tracking
- Prometheus/Grafana for monitoring

---

## 🚀 **QUICK START - DEVELOPMENT**

### **1. Install Dependencies**

```powershell
# Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### **2. Set Up Environment Variables**

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Application
APP_ENV=development
LOG_LEVEL=INFO

# CORS
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Security
JWT_SECRET=your-random-32-character-secret-key-here

# Supabase (REQUIRED for production, optional for dev)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Redis (REQUIRED for production rate limiting, optional for dev)
REDIS_URL=redis://localhost:6379/0

# AI Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Monitoring (optional)
SENTRY_DSN=https://your-sentry-dsn
```

### **3. Set Up Database**

If using Supabase:
1. Go to your Supabase project dashboard
2. Navigate to SQL Editor
3. Run the complete `database_migration.sql` script

If self-hosting PostgreSQL:
```bash
psql -U postgres -h localhost -d ai_lawyer < database_migration.sql
```

### **4. Start Redis** (if not running)

Using Docker:
```bash
docker run -d -p 6379:6379 --name redis-cache redis:7-alpine
```

Or install locally from https://redis.io/download

### **5. Start the Server**

```powershell
# Development mode
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use the ASGI entry point
uvicorn main:app --reload
```

The API will be available at:
- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

---

## 🔧 **PRODUCTION DEPLOYMENT**

### **Option A: Docker Deployment (Recommended)**

#### **1. Build Docker Image**

```bash
docker build -t ai-lawyer-backend:latest .
```

#### **2. Run with Docker Compose**

Create `docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  backend:
    image: ai-lawyer-backend:latest
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - SENTRY_DSN=${SENTRY_DSN}
    depends_on:
      - redis
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

volumes:
  redis_data:
```

Run:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### **Option B: Direct Deployment**

#### **1. Install System Dependencies**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv postgresql-client redis-server

# Install C++ build tools (required for some packages)
sudo apt-get install -y build-essential
```

#### **2. Set Up Virtual Environment**

```bash
python3.11 -m venv /opt/ai-lawyer/venv
source /opt/ai-lawyer/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### **3. Configure systemd Service**

Create `/etc/systemd/system/ai-lawyer.service`:

```ini
[Unit]
Description=AI Lawyer Backend API
After=network.target redis.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/ai-lawyer/backend
Environment="PATH=/opt/ai-lawyer/venv/bin"
ExecStart=/opt/ai-lawyer/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-lawyer
sudo systemctl start ai-lawyer
sudo systemctl status ai-lawyer
```

#### **4. Configure Nginx Reverse Proxy**

Create `/etc/nginx/sites-available/ai-lawyer`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    # Rate limiting at nginx level (additional layer)
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (for future SSE streaming)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/ai-lawyer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🗄️ **DATABASE SETUP**

### **Supabase Setup (Recommended)**

1. Create account at https://supabase.com
2. Create new project
3. Get credentials from Settings → API
4. Run `database_migration.sql` in SQL Editor

### **Self-Hosted PostgreSQL**

```bash
# Install PostgreSQL 15+
sudo apt-get install postgresql-15 postgresql-contrib-15

# Enable pgvector extension
sudo apt-get install postgresql-15-pgvector

# Create database
createdb -U postgres ai_lawyer

# Run migration
psql -U postgres -d ai_lawyer -f database_migration.sql
```

---

## 🔒 **SECURITY CHECKLIST**

Before going to production, ensure:

- [ ] JWT_SECRET is at least 32 random characters
- [ ] All default passwords changed
- [ ] HTTPS enabled (Let's Encrypt or cloud provider SSL)
- [ ] Firewall configured (only ports 80, 443, 22 open)
- [ ] Database credentials rotated
- [ ] API keys stored in secrets manager (Doppler/AWS Secrets Manager)
- [ ] Rate limiting enabled
- [ ] CORS restricted to your frontend domain only
- [ ] Sentry configured for error tracking
- [ ] Regular backups scheduled
- [ ] Monitoring alerts configured

---

## 📊 **MONITORING & OBSERVABILITY**

### **Sentry Integration**

1. Create account at https://sentry.io
2. Create new project (FastAPI)
3. Add DSN to `.env`:
   ```env
   SENTRY_DSN=https://xxx@yyy.ingest.sentry.io/zzz
   ```

### **Prometheus Metrics**

The app exposes metrics at `/metrics` (requires prometheus-client):

```yaml
# In docker-compose.yml
backend:
  ports:
    - "8000:8000"
    - "9090:9090"  # Prometheus port
```

### **Health Checks**

Configure load balancer to check:
```
GET /health
Expected response: {"status": "ok", "redis": true, "supabase": true}
```

---

## ⚡ **PERFORMANCE TUNING**

### **Uvicorn Workers**

For production, use multiple workers:

```bash
# Calculate: workers = (2 x CPU cores) + 1
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **Redis Caching**

Ensure Redis is configured for persistence:

```conf
# redis.conf
save 900 1
save 300 10
save 60 10000
maxmemory 2gb
maxmemory-policy allkeys-lru
```

---

## 🧪 **TESTING**

### **Run Tests**

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api/test_legal_endpoints.py -v
```

---

## 🆘 **TROUBLESHOOTING**

### **Port Already in Use**

```powershell
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux
lsof -ti:8000 | xargs kill -9
```

### **pyiceberg Installation Error**

This is a known issue on Windows. The app works fine without it as Supabase client doesn't require pyiceberg directly. Ignore the warning if Supabase connectivity works.

### **Redis Connection Refused**

1. Ensure Redis is running: `docker ps | grep redis`
2. Check Redis URL format: `redis://localhost:6379/0`
3. Verify firewall allows port 6379

### **Database Tables Not Found**

Run the migration script:
```bash
psql -U postgres -d ai_lawyer -f database_migration.sql
```

---

## 📈 **SCALING STRATEGY**

### **Horizontal Scaling**

1. Deploy multiple backend instances behind load balancer
2. Use managed Redis (AWS ElastiCache, Redis Cloud)
3. Use managed PostgreSQL (Supabase, AWS RDS)
4. Enable session affinity for WebSocket connections

### **Vertical Scaling**

Increase resources:
- **CPU:** More cores = more Uvicorn workers
- **RAM:** More memory = larger Redis cache
- **GPU:** Optional for faster embeddings (not required)

---

## ✅ **PRODUCTION READINESS CHECKLIST**

- [ ] All dependencies installed
- [ ] Database migrated successfully
- [ ] Redis connected and caching
- [ ] All environment variables set
- [ ] HTTPS configured
- [ ] Rate limiting working
- [ ] Health checks passing
- [ ] Error tracking (Sentry) configured
- [ ] Backups scheduled
- [ ] Monitoring alerts configured
- [ ] Load testing completed (>100 concurrent users)
- [ ] Documentation reviewed
- [ ] Team trained on operations

---

## 📞 **SUPPORT**

For issues or questions:
- GitHub Issues: [Link to repo]
- Email: support@yourcompany.com
- Documentation: https://docs.yourcompany.com

---

*Last updated: March 2026*
*Version: 2.0.0*
