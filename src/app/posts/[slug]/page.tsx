import { Paper } from "@/components/posts/paper"
import { getPostBySlug } from "@/lib/posts"
import { notFound } from "next/navigation"
import { Suspense } from "react"
import { PostHeader } from "@/components/posts/post-header"
import { PostSidebar } from "@/components/posts/post-sidebar"
import { getTableOfContents } from "@/lib/toc"
import { Metadata } from "next"
import { siteConfig } from "@/config/site"

interface PostPageProps {
    params: Promise<{
        slug: string
    }>
}

export async function generateMetadata({ params }: PostPageProps): Promise<Metadata> {
    const resolvedParams = await params
    const post = await getPostBySlug(resolvedParams.slug, false)
    if (!post) return {}

    const ogImage = post.coverImage || siteConfig.ogImage
    return {
        title: post.title,
        description: post.excerpt,
        authors: [{ name: post.author }],
        openGraph: {
            title: post.title,
            description: post.excerpt,
            type: 'article',
            url: `${siteConfig.url}/posts/${resolvedParams.slug}`,
            images: [{ url: ogImage, width: 1200, height: 630, alt: post.title }],
            publishedTime: post.date,
            authors: [post.author],
            tags: post.tags,
        },
        twitter: {
            card: 'summary_large_image',
            title: post.title,
            description: post.excerpt,
            images: [ogImage],
        },
    }
}

export default async function PostPage({ params }: PostPageProps) {
    const resolvedParams = await params
    const post = await getPostBySlug(resolvedParams.slug, false)

    if (!post) {
        notFound()
    }

    const tableOfContents = await getTableOfContents(post.content)

    return (
        <>
            <PostHeader />
            <div className="container relative py-6 md:py-10">
                <div className="flex flex-col lg:flex-row lg:gap-10">
                    <div className="flex-1">
                        <Suspense fallback={<div>加载中...</div>}>
                            <Paper post={post} />
                        </Suspense>
                    </div>
                    <div className="lg:w-64">
                        <PostSidebar tableOfContents={tableOfContents} />
                    </div>
                </div>
            </div>
        </>
    )
} 