import resend
from fastapi import HTTPException, status
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

async def send_reset_password_email(email: str, code: str, expiry: int) -> bool:
    """
    Gửi email đặt lại mật khẩu sử dụng dịch vụ Brevo.

    Args:
        email (str): Địa chỉ email của người dùng.
        code (str): Token đặt lại mật khẩu.
        expiry_hours (int): Thời gian hết hạn của link (giờ).

    Returns:
        bool: True nếu gửi email thành công.

    Raises:
        HTTPException: Nếu có lỗi khi gửi email.
    """
    try:

        resend.api_key = os.environ.get('RESEND_API_KEY')
        # Email content
        subject = "Mã xác minh đăng nhập"
        sender = "onboarding@resend.dev"
        reciver = ["hunterdev03@gmail.com"]
        # Create reset link
        # reset_link = f"https://yourapp.com/reset-password?token={reset_token}"


        # Email content
        subject = "Yêu cầu đặt lại mật khẩu cho tài khoản của bạn"
        html_content = f"""
        <html>
            <body>
                <h2>Xin chào {reciver[0].split('@')[0]},</h2>
                <p>Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản của bạn.</p>
                <p>Mã xác minh để đặt lại mật khẩu. Mã này sẽ hết hạn sau {expiry} phut: <strong>{code}</strong></p>

                <p>Nếu bạn không yêu cầu đặt lại mật khẩu, vui lòng bỏ qua email này hoặc liên hệ với chúng tôi qua <a href="mailto:support@yourdomain.com">support@yourdomain.com</a>.</p>
                <p>Trân trọng,<br>Đội ngũ Angle Lawer</p>
            </body>
        </html>
        """

        # Create email payload
        params: resend.Emails.SendParams = {
            "from": sender,
            "to": reciver,
            "subject": subject,
            "html": html_content,
        }
        # Create and send email
        email: resend.Email = resend.Emails.send(params)
        return True
    except resend.errors.ApiError as e:
        logger.error(f"Lỗi khi gửi email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
        )
    except Exception as e:
        logger.error(f"Lỗi hệ thống khi gửi email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
        )


async def send_verification_code_email(email: str, code: str, expiry_minutes: int = 10) -> bool:
    """
    Gửi email chứa mã xác minh đăng nhập sử dụng dịch vụ Brevo.

    Args:
        email (str): Địa chỉ email của người dùng.
        code (str): Mã xác minh 6 chữ số.
        expiry_minutes (int): Thời gian hết hạn của mã (phút).

    Returns:
        bool: True nếu gửi email thành công.

    Raises:
        HTTPException: Nếu có lỗi khi gửi email.
    """
    try:
        resend.api_key = os.environ.get('RESEND_API_KEY')

        # Email content
        subject = "Mã xác minh đăng nhập"
        sender = "onboarding@resend.dev"
        reciver = [email]
        html_content = f"""
        <html>
            <body>
                <h2>Xin chào {reciver[0].split('@')[0]},</h2>
                <p>Mã xác minh đăng nhập của bạn là: <strong>{code}</strong></p>
                <p>Mã này sẽ hết hạn sau {expiry_minutes} phút.</p>
                <p>Nếu bạn không yêu cầu đăng nhập, vui lòng bỏ qua email này hoặc liên hệ với chúng tôi qua <a href="mailto:support@yourdomain.com">support@yourdomain.com</a>.</p>
                <p>Trân trọng,<br>Đội ngũ Angel Lawer</p>
            </body>
        </html>
        """
        params: resend.Emails.SendParams = {
            "from": sender,
            "to": reciver,
            "subject": subject,
            "html": html_content,
        }
        # Create and send email
        email: resend.Email = resend.Emails.send(params)
        return True
    except resend.errors.ApiError as e:
        logger.error(f"Lỗi khi gửi email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
        )
    except Exception as e:
        logger.error(f"Lỗi hệ thống khi gửi email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
    )


async def send_suspicious_activity_email(email: str) -> bool:
    """
    Gửi email thông báo hoạt động đáng ngờ (ví dụ: refresh token không hợp lệ).

    Args:
        email (str): Địa chỉ email của người dùng.

    Returns:
        bool: True nếu gửi email thành công.

    Raises:
        HTTPException: Nếu có lỗi khi gửi email.
    """
    try:
        resend.api_key = os.environ.get('RESEND_API_KEY')


        subject = "Cảnh báo hoạt động đáng ngờ"
        # Email content
        subject = "Mã xác minh đăng nhập"
        sender = "onboarding@resend.dev"
        reciver = [email]
        html_content = f"""
        <html>
            <body>
                <h2>Xin chào {reciver[0].split('@')[0]},</h2>
                <p>Chúng tôi đã phát hiện một nỗ lực sử dụng refresh token không hợp lệ để truy cập tài khoản của bạn.</p>
                <p>Nếu đây không phải là bạn, vui lòng bảo mật tài khoản ngay lập tức và liên hệ với chúng tôi qua <a href="mailto:support@yourdomain.com">support@yourdomain.com</a>.</p>
                <p>Trân trọng,<br>Đội ngũ Angle Lawer</p>
            </body>
        </html>
        """

        params: resend.Emails.SendParams = {
            "from": sender,
            "to": reciver,
            "subject": subject,
            "html": html_content,
        }
        # Create and send email
        email: resend.Email = resend.Emails.send(params)
        return True

    except resend.errors.ApiError as e:
        logger.error(f"Lỗi khi gửi email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi hệ thống. Vui lòng thử lại sau."
        )
    except Exception as e:
        logger.error(f"Lỗi hệ thống khi gửi email cảnh báo: {str(e)}")
        return False