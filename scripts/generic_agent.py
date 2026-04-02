from livekit.agents import Agent
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit import api
import logging

logger = logging.getLogger(__name__)

class GenericAgent(Agent):
    # Tự động chào khi vào phòng
    # async def on_enter(self):
    #     self.session.generate_reply()
    
    # Tự động kết thúc cuộc trò chuyện khi người dùng yêu cầu
    @function_tool
    async def end_conversation(self):
        """Gọi hàm này khi người dùng muốn kết thúc cuộc trò chuyện"""
        self.session.interrupt()

        await self.session.generate_reply(
            instructions=f"tạm biệt", allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

    async def on_enter(self):
        """Called when the agent becomes active."""
        logger.info("Agent session started")
        
        # Generate initial greeting
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )
    
    async def on_exit(self):
        """Called when the agent session ends."""
        logger.info("Agent session ended")
