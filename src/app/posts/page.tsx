import { Metadata } from "next"
import { siteConfig } from "@/config/site"
import { getAllPosts, getCategories } from "@/lib/posts"
import { PostList } from "@/components/index/post-list"
import { i18n } from "@/config/i18n"

export const metadata: Metadata = {
    title: {
        default: i18n.zh.category.title,
        template: `%s | ${siteConfig.name}`
    },
    description: i18n.zh.category.description,
    openGraph: {
        title: `${i18n.zh.category.title} | ${siteConfig.name}`,
        description: i18n.zh.category.description,
        url: `${siteConfig.url}/posts`,
    },
}

export default async function PostsPage() {
    const [posts, categories] = await Promise.all([
        getAllPosts(false),
        getCategories()
    ])

    return (
        <div className="container">
            <PostList posts={posts} categories={categories} />
        </div>
    )
} 