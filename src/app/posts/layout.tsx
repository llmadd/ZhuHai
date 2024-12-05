import { Metadata } from "next"

export const metadata: Metadata = {
    title: '博客文章 | My Blog',
    description: '分享技术文章和生活感悟',
}

export default function PostsLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
            {children}
        </div>
    )
} 