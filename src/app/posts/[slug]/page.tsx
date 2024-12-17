import { Metadata } from "next"
import { getPostBySlug } from "@/lib/posts"
import { Paper } from "@/components/posts/paper"
import { notFound } from "next/navigation"
import { siteConfig } from "@/config/site"

interface PostPageProps {
    params: {
        slug: string
    }
}

export async function generateMetadata({ params }: PostPageProps): Promise<Metadata> {
    const post = await getPostBySlug(params.slug, false)
    if (!post) return {}

    return {
        title: post.title.zh,
        description: post.excerpt.zh,
        openGraph: {
            title: post.title.zh,
            description: post.excerpt.zh,
            type: 'article',
            url: `${siteConfig.url}/posts/${post.slug}`,
            images: post.coverImage ? [
                {
                    url: post.coverImage,
                    width: 1200,
                    height: 630,
                    alt: post.coverImageAlt?.zh || post.title.zh,
                }
            ] : undefined,
        },
    }
}

export default async function PostPage({ params }: PostPageProps) {
    const post = await getPostBySlug(params.slug, false)

    if (!post) {
        notFound()
    }

    return <Paper post={post} />
} 