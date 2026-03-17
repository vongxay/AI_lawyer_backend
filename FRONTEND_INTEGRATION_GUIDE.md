# 🚀 Frontend Integration Guide

**Quick start guide for connecting React frontend to AI Lawyer FastAPI backend**

---

## 1. Environment Setup

### Backend Configuration

Create `.env` file in backend root:

```env
# ── App Settings ──────────────────────────────────────────────────────────────
APP_NAME=AI Lawyer Backend
APP_VERSION=2.0.0
APP_ENV=development
LOG_LEVEL=DEBUG

# ── CORS (Add your frontend URL here) ─────────────────────────────────────────
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# ── Database ──────────────────────────────────────────────────────────────────
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# ── LLM APIs ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...

# ── Redis (Optional for development) ──────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Security ──────────────────────────────────────────────────────────────────
JWT_SECRET=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=60

# ── Production Monitoring ─────────────────────────────────────────────────────
# SENTRY_DSN=https://your-sentry-dsn
```

---

## 2. Frontend API Client Setup

### TypeScript API Client

```typescript
// src/lib/api-client.ts

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface LegalQueryRequest {
  question: string;
  case_id?: string;
  jurisdiction?: string;
}

export interface Statute {
  name: string;
  section: string;
  text: string;
  status: string;
  year?: number;
}

export interface Precedent {
  case_no: string;
  court?: string;
  relevance?: string;
  outcome?: string;
  graph_path?: string;
}

export interface CitationItem {
  ref: string;
  status: 'VERIFIED' | 'OUTDATED' | 'UNVERIFIED' | 'REJECTED';
  note?: string;
  db_match?: string;
  year?: number;
  reason?: string;
  source_links: string[];
}

export interface LegalQueryResponse {
  irac: {
    issue: {
      primary: string;
      secondary: string[];
    };
    rule: {
      statutes: Statute[];
      precedents: Precedent[];
    };
    application: {
      analysis: string;
      strengths: string[];
      weaknesses: string[];
      counter_args: string[];
      rebuttals: string[];
    };
    conclusion: {
      recommendation: string;
      action_steps: string[];
      risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
      win_probability: number;
      settlement_note?: string;
    };
  };
  citations: CitationItem[];
  citations_verified: boolean;
  confidence: number;
  agents_used: string[];
  processing_time_ms: number;
  escalated_to_expert: boolean;
  disclaimer: string;
}

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  setToken(token: string) {
    this.token = token;
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    
    return headers;
  }

  // ── Legal Query Endpoints ───────────────────────────────────────────────────

  async queryLegal(payload: LegalQueryRequest): Promise<LegalQueryResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/legal/query`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query failed');
    }

    return response.json();
  }

  // ── SSE Streaming Query ─────────────────────────────────────────────────────

  streamLegalQuery(
    payload: LegalQueryRequest,
    onEvent: (event: { type: string; message?: string; data?: any }) => void,
    onError?: (error: Error) => void
  ): () => void {
    const eventSource = new EventSource(
      `${this.baseUrl}/api/v1/legal/query/stream`,
      {
        headers: this.token ? { 'Authorization': `Bearer ${this.token}` } : {},
      }
    );

    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      onEvent(parsed);
      
      if (parsed.type === 'done') {
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      onError?.(new Error('Stream connection lost'));
    };

    // Return cleanup function
    return () => eventSource.close();
  }

  // ── Case Memory Endpoints ───────────────────────────────────────────────────

  async getCaseMemory(caseId: string) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/memory/case/${caseId}`,
      { headers: this.getHeaders() }
    );
    return response.json();
  }

  async getCaseTimeline(caseId: string) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/memory/case/${caseId}/timeline`,
      { headers: this.getHeaders() }
    );
    return response.json();
  }

  // ── Document Analysis ───────────────────────────────────────────────────────

  async analyzeDocument(file: File, question?: string): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    if (question) formData.append('question', question);

    const response = await fetch(
      `${this.baseUrl}/api/v1/documents/analyze`,
      {
        method: 'POST',
        headers: {
          // Don't set Content-Type - browser will set it with boundary
          ...this.token ? { 'Authorization': `Bearer ${this.token}` } : {},
        },
        body: formData,
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
  }

  // ── Evidence Analysis ───────────────────────────────────────────────────────

  async analyzeEvidence(files: File[], question?: string): Promise<any> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    if (question) formData.append('question', question);

    const response = await fetch(
      `${this.baseUrl}/api/v1/evidence/analyze`,
      {
        method: 'POST',
        headers: {
          ...this.token ? { 'Authorization': `Bearer ${this.token}` } : {},
        },
        body: formData,
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Evidence analysis failed');
    }

    return response.json();
  }

  // ── Health Check ────────────────────────────────────────────────────────────

  async checkHealth(): Promise<{ status: string; version: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Export singleton instance
export const apiClient = new ApiClient(API_BASE_URL);
```

---

## 3. React Hook Examples

### Legal Query Hook

```typescript
// src/hooks/useLegalQuery.ts

import { useState, useCallback } from 'react';
import { apiClient, LegalQueryRequest, LegalQueryResponse } from '@/lib/api-client';

export function useLegalQuery() {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<LegalQueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const query = useCallback(async (payload: LegalQueryRequest) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiClient.queryLegal(payload);
      setResponse(result);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, response, error, query };
}
```

### SSE Streaming Hook

```typescript
// src/hooks/useStreamingQuery.ts

import { useState, useCallback, useRef, useEffect } from 'react';
import { apiClient, LegalQueryRequest } from '@/lib/api-client';

interface StreamEvent {
  type: 'status' | 'result' | 'thinking' | 'done';
  message?: string;
  data?: any;
}

export function useStreamingQuery() {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  const stream = useCallback(async (payload: LegalQueryRequest) => {
    setLoading(true);
    setError(null);
    setEvents([]);

    try {
      cleanupRef.current = apiClient.streamLegalQuery(
        payload,
        (event) => {
          setEvents(prev => [...prev, event]);
          
          if (event.type === 'done') {
            setLoading(false);
          }
        },
        (err) => {
          setError(err.message);
          setLoading(false);
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Stream failed');
      setLoading(false);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, []);

  return { events, loading, error, stream };
}
```

---

## 4. Example Components

### Chat Interface Component

```tsx
// src/components/chat/ChatWindow.tsx

import React, { useState } from 'react';
import { useLegalQuery } from '@/hooks/useLegalQuery';
import { IRACView } from './IRACView';
import { CitationBadges } from './CitationBadges';

export function ChatWindow() {
  const [question, setQuestion] = useState('');
  const { loading, response, query } = useLegalQuery();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    await query({
      question: question.trim(),
      jurisdiction: 'TH',
    });

    setQuestion('');
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading && (
          <div className="text-blue-600 animate-pulse">
            กำลังวิเคราะห์คำถามตามหลัก IRAC...
          </div>
        )}

        {response && (
          <div className="space-y-4">
            {/* IRAC Analysis */}
            <IRACView irac={response.irac} />

            {/* Citations */}
            <CitationBadges citations={response.citations} />

            {/* Confidence & Metadata */}
            <div className="text-sm text-gray-500">
              <span>ความมั่นใจ: {(response.confidence * 100).toFixed(0)}%</span>
              <span className="mx-2">•</span>
              <span>เวลาประมวลผล: {response.processing_time_ms}ms</span>
            </div>

            {/* Disclaimer */}
            <div className="text-xs text-gray-400 italic">
              {response.disclaimer}
            </div>
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="border-t p-4">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="ถามคำถามกฎหมายที่นี่..."
          className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
          rows={3}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="mt-2 px-6 py-2 bg-blue-600 text-white rounded-lg 
                     hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'กำลังวิเคราะห์...' : 'ส่งคำถาม'}
        </button>
      </form>
    </div>
  );
}
```

### IRAC Display Component

```tsx
// src/components/chat/IRACView.tsx

interface IRACViewProps {
  irac: {
    issue: { primary: string; secondary: string[] };
    rule: { statutes: any[]; precedents: any[] };
    application: { analysis: string; strengths: string[]; weaknesses: string[] };
    conclusion: { recommendation: string; action_steps: string[] };
  };
}

export function IRACView({ irac }: IRACViewProps) {
  return (
    <div className="space-y-6">
      {/* Issue */}
      <section className="bg-red-50 p-4 rounded-lg">
        <h3 className="font-bold text-lg text-red-800 mb-2">
          🔴 ประเด็นกฎหมาย (Issue)
        </h3>
        <p className="text-gray-800">{irac.issue.primary}</p>
        {irac.issue.secondary.length > 0 && (
          <ul className="mt-2 space-y-1 text-sm text-gray-600">
            {irac.issue.secondary.map((item, i) => (
              <li key={i}>• {item}</li>
            ))}
          </ul>
        )}
      </section>

      {/* Rule */}
      <section className="bg-blue-50 p-4 rounded-lg">
        <h3 className="font-bold text-lg text-blue-800 mb-2">
          ⚖️ หลักกฎหมาย (Rule)
        </h3>
        <div className="space-y-3">
          {irac.rule.statutes.map((statute, i) => (
            <div key={i} className="text-sm">
              <span className="font-semibold">{statute.name} {statute.section}</span>
              <p className="text-gray-700 mt-1">{statute.text}</p>
            </div>
          ))}
          {irac.rule.precedents.map((precedent, i) => (
            <div key={i} className="text-sm">
              <span className="font-semibold">{precedent.case_no}</span>
              <span className="text-gray-600 ml-2">({precedent.court})</span>
              <p className="text-gray-700 mt-1">{precedent.relevance}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Application */}
      <section className="bg-yellow-50 p-4 rounded-lg">
        <h3 className="font-bold text-lg text-yellow-800 mb-2">
          📊 การวิเคราะห์ (Application)
        </h3>
        <p className="text-gray-800 mb-3">{irac.application.analysis}</p>
        
        <div className="grid grid-cols-2 gap-3">
          <div>
            <h4 className="font-semibold text-green-700 text-sm">✓ จุดแข็ง</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              {irac.application.strengths.map((item, i) => (
                <li key={i}>• {item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-red-700 text-sm">✗ จุดอ่อน</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              {irac.application.weaknesses.map((item, i) => (
                <li key={i}>• {item}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* Conclusion */}
      <section className="bg-green-50 p-4 rounded-lg">
        <h3 className="font-bold text-lg text-green-800 mb-2">
          ✅ สรุปและคำแนะนำ (Conclusion)
        </h3>
        <p className="text-gray-800 font-medium">{irac.conclusion.recommendation}</p>
        <ol className="mt-3 space-y-1 text-sm text-gray-700 list-decimal list-inside">
          {irac.conclusion.action_steps.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
        <div className="mt-3 text-sm">
          <span className="text-gray-600">โอกาสชนะคดี: </span>
          <span className="font-bold text-green-700">
            {(irac.conclusion.win_probability * 100).toFixed(0)}%
          </span>
        </div>
      </section>
    </div>
  );
}
```

---

## 5. Testing the Integration

### Test Script

```typescript
// test-integration.ts

import { apiClient } from './api-client';

async function testBackend() {
  console.log('🧪 Testing AI Lawyer Backend Integration...\n');

  // Test 1: Health Check
  try {
    const health = await apiClient.checkHealth();
    console.log('✅ Health Check:', health);
  } catch (error) {
    console.error('❌ Health Check Failed:', error);
    return;
  }

  // Test 2: Legal Query
  try {
    const response = await apiClient.queryLegal({
      question: 'สัญญาจ้างแรงงานต้องทำเป็นหนังสือหรือไม่',
      jurisdiction: 'TH',
    });
    
    console.log('\n✅ Legal Query Response:');
    console.log('- Issue:', response.irac.issue.primary);
    console.log('- Statutes:', response.irac.rule.statutes.length);
    console.log('- Confidence:', response.confidence);
    console.log('- Processing Time:', response.processing_time_ms, 'ms');
  } catch (error) {
    console.error('❌ Legal Query Failed:', error);
  }

  // Test 3: Case Memory (if case exists)
  try {
    const memory = await apiClient.getCaseMemory('test-case-id');
    console.log('\n✅ Case Memory:', memory);
  } catch (error) {
    console.log('\n⚠️ Case Memory Not Found (expected for new DB)');
  }

  console.log('\n✨ Integration tests complete!\n');
}

testBackend();
```

---

## 6. Common Issues & Solutions

### Issue 1: CORS Errors

**Error:** `Access to fetch at 'http://localhost:8000' has been blocked by CORS policy`

**Solution:** Update `ALLOWED_ORIGINS` in backend `.env`:
```env
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

Then restart backend server.

---

### Issue 2: 401 Unauthorized

**Error:** `401 Unauthorized` on all authenticated endpoints

**Solution:** Ensure JWT token is attached:
```typescript
apiClient.setToken('your-jwt-token-here');
```

Or use optional auth endpoints that don't require login.

---

### Issue 3: WebSocket Connection Fails

**Error:** SSE stream doesn't connect

**Solution:** Check Nginx/proxy configuration for SSE support:
```nginx
location /api/v1/legal/query/stream {
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_cache off;
}
```

---

### Issue 4: Large File Upload Fails

**Error:** `File size exceeds limit`

**Solution:** Increase `MAX_UPLOAD_SIZE_MB` in backend config or compress files before upload.

---

## 7. Next Steps

### Week 1: Basic Integration
- [ ] Set up API client
- [ ] Implement chat UI
- [ ] Connect legal query endpoint
- [ ] Display IRAC responses

### Week 2: Advanced Features
- [ ] Add SSE streaming
- [ ] Implement case memory panel
- [ ] Build citation visualization

### Week 3: Document Handling
- [ ] File upload component
- [ ] Document analysis display
- [ ] Evidence gallery

### Week 4: Polish & Optimize
- [ ] Loading states
- [ ] Error handling
- [ ] Performance optimization
- [ ] User testing

---

**Happy Integrating! 🚀**

For detailed API documentation, visit: http://localhost:8000/docs
