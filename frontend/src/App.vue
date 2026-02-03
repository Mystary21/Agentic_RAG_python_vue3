<script setup lang="ts">
import { ref, nextTick } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { Send, Image as ImageIcon, Loader2, XCircle } from 'lucide-vue-next';

// --- Type Definitions ---
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  image?: string; // Base64 string for preview
}

// --- State ---
const userInput = ref('');
const messages = ref<Message[]>([
  { role: 'system', content: 'Hello! I am your AI Agent. I can help you search documents, analyze data, or read images.' }
]);
const isLoading = ref(false);
const selectedImage = ref<string | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const chatContainer = ref<HTMLElement | null>(null);

// --- Methods ---

// 1. Handle Image Selection
const triggerFileUpload = () => {
  fileInput.value?.click();
};

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files[0]) {
    const file = target.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      selectedImage.value = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }
};

const clearImage = () => {
  selectedImage.value = null;
  if (fileInput.value) fileInput.value.value = '';
};

// 2. Render Markdown Safely
const renderMarkdown = (text: string) => {
  const rawHtml = marked.parse(text) as string;
  return DOMPurify.sanitize(rawHtml);
};

// 3. Scroll to Bottom
const scrollToBottom = async () => {
  await nextTick();
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
};

// 4. Send Message (The Core Logic)
const sendMessage = async () => {
  if ((!userInput.value.trim() && !selectedImage.value) || isLoading.value) return;

  // Prepare Payload
  const currentQuery = userInput.value;
  const currentImage = selectedImage.value;
  
  // Add User Message to UI
  messages.value.push({
    role: 'user',
    content: currentQuery,
    image: currentImage || undefined
  });

  // Clear Input
  userInput.value = '';
  selectedImage.value = null; // Image is sent, clear buffer
  isLoading.value = true;
  await scrollToBottom();

  // Create Placeholder for Assistant Response
  const assistantMessageIndex = messages.value.push({
    role: 'assistant',
    content: ''
  }) - 1;

  try {
    // Call Python Backend
    // Note: URL is relative. In Docker, Nginx/FastAPI serves this.
    // In Dev, Vite proxy handles it.
    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: currentQuery,
        history: messages.value.slice(0, -2), // Send context excluding current turn
        image_data: currentImage
      })
    });

    if (!response.ok) throw new Error('Network response was not ok');
    if (!response.body) throw new Error('ReadableStream not supported');

    // Handle Streaming (SSE)
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      // Append text directly. Since the backend sends raw tokens in SSE format,
      // you might need to strip "data: " prefix depending on strict SSE compliance.
      // Assuming your Python 'yield token' sends raw text chunks:
      messages.value[assistantMessageIndex].content += chunk;
      
      scrollToBottom();
    }

  } catch (error) {
    console.error('Chat Error:', error);
    messages.value[assistantMessageIndex].content += '\n\n**[System Error]**: Unable to connect to the agent.';
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="flex flex-col h-screen max-w-4xl mx-auto bg-white shadow-xl overflow-hidden">
    
    <header class="bg-slate-900 text-white p-4 flex items-center justify-between">
      <h1 class="text-xl font-bold flex items-center gap-2">
        🤖 Agentic RAG
        <span class="text-xs font-normal bg-slate-700 px-2 py-1 rounded">A100 Powered</span>
      </h1>
    </header>

    <div ref="chatContainer" class="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50">
      <div v-for="(msg, index) in messages" :key="index" 
           :class="['flex', msg.role === 'user' ? 'justify-end' : 'justify-start']">
        
        <div :class="['max-w-[85%] rounded-2xl p-4 shadow-sm', 
                      msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200 text-gray-800']">
          
          <div v-if="msg.image" class="mb-3">
            <img :src="msg.image" alt="Uploaded context" class="max-h-64 rounded-lg border border-white/20" />
          </div>

          <div v-if="msg.role === 'assistant'" 
               class="prose prose-sm max-w-none dark:prose-invert"
               v-html="renderMarkdown(msg.content)">
          </div>
          <div v-else class="whitespace-pre-wrap">{{ msg.content }}</div>
        
        </div>
      </div>
    </div>

    <div class="p-4 bg-white border-t border-gray-200">
      
      <div v-if="selectedImage" class="mb-2 flex items-center gap-2 bg-blue-50 p-2 rounded-lg inline-block">
        <span class="text-xs text-blue-600 font-medium">Image attached</span>
        <button @click="clearImage" class="text-blue-400 hover:text-blue-600">
          <XCircle class="w-4 h-4" />
        </button>
      </div>

      <div class="flex gap-2 items-end">
        <button @click="triggerFileUpload" 
                class="p-3 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-full transition"
                title="Upload Image">
          <ImageIcon class="w-6 h-6" />
        </button>
        <input type="file" ref="fileInput" @change="handleFileChange" accept="image/*" class="hidden" />

        <textarea 
          v-model="userInput" 
          @keydown.enter.prevent="sendMessage"
          placeholder="Ask something or upload a chart..." 
          class="flex-1 p-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none h-14 bg-gray-50"
        ></textarea>

        <button @click="sendMessage" 
                :disabled="isLoading || (!userInput && !selectedImage)"
                class="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center min-w-[3rem]">
          <Loader2 v-if="isLoading" class="w-6 h-6 animate-spin" />
          <Send v-else class="w-5 h-5" />
        </button>
      </div>
    </div>

  </div>
</template>
