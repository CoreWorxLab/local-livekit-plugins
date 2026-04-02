from datetime import datetime
from livekit.agents.llm import function_tool

class AssistantFnc:
    """Helper functions for the assistant."""

    @function_tool(description="Hàm này trả về ngày giờ hiện tại.")
    async def get_datetime(self) -> str:
        """Trả về ngày giờ hiện tại ở định dạng dễ đọc."""
        now = datetime.now()
        # Format: Thứ Năm, 26 tháng 03 năm 2026, 16:00:39
        # In Vietnamese
        days = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        day_str = days[now.weekday()]
        return f"{day_str}, {now.strftime('%d/%m/%Y %H:%M:%S')}"

    @function_tool(description="Quét mã QR để thực hiện check-in hoặc check-out.")
    async def scan_qr_code(self, operation: str) -> str:
        """Quét mã QR. operation có thể là 'check-in' hoặc 'check-out'."""
        # Simulated QR scan
        return f"Mã QR đã được quét thành công. Đã thực hiện {operation} cho khách hàng."

    @function_tool(description="Quét Căn cước công dân (CCCD) để lấy thông tin định danh.")
    async def scan_cccd(self) -> str:
        """Quét thẻ CCCD và trả về thông tin cơ bản."""
        # Simulated CCCD scan
        return "Đã quét CCCD thành công. Họ tên: Nguyễn Văn A, Số CCCD: 012345678901, Địa chỉ: Hà Nội."

    @function_tool(description="Đăng ký thông tin khách đến thăm công ty.")
    async def register_visitor(self, name: str, purpose: str, contact: str) -> str:
        """Đăng ký khách thăm. Cần tên khách, mục đích và thông tin liên lạc."""
        # Simulated registration
        return f"Đã đăng ký thành công cho khách {name}. Mục đích: {purpose}. Chúng tôi sẽ liên hệ qua {contact}."

    @function_tool(description="Tìm kiếm thông tin thời gian thực từ internet.")
    async def search_web(self, query: str) -> str:
        """Tìm kiếm thông tin từ internet cho các câu hỏi về thời sự, thời tiết, kiến thức mới."""
        # Simulated web search
        return f"Kết quả tìm kiếm cho '{query}': Đây là thông tin cập nhật mới nhất từ internet về vấn đề này."

    @function_tool(description="Gọi điện cho nhân viên công ty (ví dụ: anh Huy) để lấy thông tin hoặc xác nhận.")
    async def call_staff(self, staff_name: str, reason: str) -> str:
        """Thực hiện cuộc gọi cho nhân viên. Ví dụ: 'anh Huy'."""
        # Simulated human-in-the-loop call
        if "Huy" in staff_name:
            return f"Đang kết nối với anh Huy... Anh Huy xác nhận: 'Cứ cho khách vào, tôi đang đợi ở phòng họp số 1'."
        return f"Đang gọi cho {staff_name}... Hiện tại nhân viên không nhấc máy, vui lòng thử lại sau."

    @function_tool(description="Kiểm tra nhận diện khuôn mặt (Face ID) để đảm bảo không có giả mạo.")
    async def verify_face_id(self) -> str:
        """Thực hiện kiểm tra Face ID."""
        # Simulated Face ID check
        return "Face ID trùng khớp. Xác nhận danh tính thành công. Có thể tiếp tục công việc."
