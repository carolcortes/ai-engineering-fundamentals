import { AIChatAgent } from '@cloudflare/ai-chat';
import { convertToModelMessages } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { streamAgent } from './agent-core';
import { compactHistory } from './context/compaction';

interface Env extends Cloudflare.Env {
  OPENAI_API_KEY: string;
  TAVILY_API_KEY: string;
  UPSTASH_VECTOR_REST_URL: string;
  UPSTASH_VECTOR_REST_TOKEN: string;
}

const SYSTEM_PROMPT = `You are a diagram design assistant. You help users create and modify diagrams on an Excalidraw canvas.

When the user asks you to create a diagram, use the generateDiagram tool to produce Excalidraw elements.

Guidelines for generating diagrams:
- Give each element a unique id (e.g. "rect-1", "text-1", "arrow-1")
- Position elements with reasonable spacing (at least 20px gap between elements)
- Use rectangles for boxes/containers, ellipses for circles, diamonds for decision points
- Add text labels inside or near shapes
- Connect related elements with arrows
- Use a clean layout: left to right or top to bottom
- Default to strokeColor "#1e1e1e" and backgroundColor "transparent"
- Set roughness to 1 for a hand-drawn look

When the user asks to modify an element, use the modifyDiagram tool with the element's id.`;

export class DesignAgent extends AIChatAgent<Env> {
  async onChatMessage() {
    const openai = createOpenAI({ apiKey: this.env.OPENAI_API_KEY });
    const model = openai('gpt-5.4-mini');

    // Compact older history if the conversation has gotten long. The recent
    // few turns stay verbatim; everything older is collapsed into one
    // summary system message.
    const allMessages = await convertToModelMessages(this.messages);
    const messages = await compactHistory(allMessages, { model });

    const result = streamAgent({
      model,
      system: SYSTEM_PROMPT,
      messages,
      env: {
        TAVILY_API_KEY: this.env.TAVILY_API_KEY,
        UPSTASH_VECTOR_REST_URL: this.env.UPSTASH_VECTOR_REST_URL,
        UPSTASH_VECTOR_REST_TOKEN: this.env.UPSTASH_VECTOR_REST_TOKEN,
      },
    });

    return result.toUIMessageStreamResponse();
  }
}
