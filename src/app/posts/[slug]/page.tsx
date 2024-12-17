import type { Metadata } from "next"
import { getPostBySlug } from "@/lib/posts"
import { Paper } from "@/components/posts/paper"
import { notFound } from "next/navigation"
import { siteConfig } from "@/config/site"

// 定义参数类型
type PageParams = {
    slug: string
}

// 定义 Props 类型
type Props = {
    params: Promise<PageParams>
    searchParams?: { [key: string]: string | string[] | undefined }
}

export async function generateMetadata(
    props: Props
): Promise<Metadata> {
    const { slug } = await props.params
    const post = await getPostBySlug(slug, false)
    if (!post) return {}

    return {
        title: post.title.zh,
        description: post.excerpt.zh,
        openGraph: {
            title: post.title.zh,
            description: post.excerpt.zh,
            type: 'article',
            url: `${siteConfig.url}/posts/${slug}`,
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

export default async function Page(props: Props) {
    const { slug } = await props.params
    const post = await getPostBySlug(slug, false)
    if (!post) notFound()

    return <Paper post={post} />
}

// 生成静态参数
export async function generateStaticParams(): Promise<PageParams[]> {
    return []
} 