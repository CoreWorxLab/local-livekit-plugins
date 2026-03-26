from datetime import datetime
from livekit.agents.llm import function_tool

class AssistantFnc:
    """Helper functions for the assistant."""

    @function_tool(description="Hàm này trả về ngày giờ hiện tại.")
    def get_datetime(self) -> str:
        """Trả về ngày giờ hiện tại ở định dạng dễ đọc."""
        now = datetime.now()
        # Format: Thứ Năm, 26 tháng 03 năm 2026, 16:00:39
        # In Vietnamese
        days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        day_str = days[now.weekday()]
        return f"{day_str}, {now.strftime('%d/%m/%Y %H:%M:%S')}"
