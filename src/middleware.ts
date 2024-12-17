import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { geolocation } from '@vercel/functions'

export function middleware(request: NextRequest) {
    // 获取国家代码
    const { country } = geolocation(request)

    // 创建响应
    const response = NextResponse.next()

    // 设置 Cookie 存储语言偏好
    if (!request.cookies.has('locale')) {
        response.cookies.set('locale', country === 'CN' ? 'zh' : 'en')
    }

    return response
}

export const config = {
    matcher: '/:path*',
} 