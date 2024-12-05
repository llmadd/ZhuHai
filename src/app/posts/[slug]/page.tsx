import { Paper } from "@/components/posts/paper"
import { getPostBySlug } from "@/lib/posts"
import { getComments } from "@/lib/comments"
import { notFound } from "next/navigation"
import { CommentForm } from "@/components/posts/comment-form"
import { CommentList } from "@/components/posts/comment-list"
import { revalidatePath } from "next/cache"
import { CommentFormData } from "@/types/comment"
import { Suspense } from "react"
import { PostHeader } from "@/components/posts/post-header"
import { PostSidebar } from "@/components/posts/post-sidebar"
import { getTableOfContents } from "@/lib/toc"

interface PostPageProps {
    params: {
        slug: string
    }
}

export default async function PostPage({ params }: PostPageProps) {
    const slug = (await Promise.resolve(params)).slug

    const [post, comments] = await Promise.all([
        getPostBySlug(slug, false),
        getComments(slug)
    ])

    if (!post) {
        notFound()
    }

    const tableOfContents = await getTableOfContents(post.content)

    async function addComment(data: CommentFormData) {
        'use server'
        try {
            const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
            const response = await fetch(`${baseUrl}/api/posts/${slug}/comments`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data })
            })

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            revalidatePath(`/posts/${slug}`)
        } catch (error) {
            console.error('Error adding comment:', error)
            throw error
        }
    }

    async function replyToComment(commentId: string, data: CommentFormData) {
        'use server'
        try {
            const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
            const response = await fetch(`${baseUrl}/api/posts/${slug}/comments`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data, parentId: commentId })
            })

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            revalidatePath(`/posts/${slug}`)
        } catch (error) {
            console.error('Error adding reply:', error)
            throw error
        }
    }

    return (
        <>
            <PostHeader />
            <div className="container relative py-6 md:py-10">
                <div className="flex flex-col lg:flex-row lg:gap-10">
                    <div className="flex-1">
                        <Suspense fallback={<div>加载中...</div>}>
                            <Paper post={post} />

                            <div className="mt-8 space-y-8 md:mt-10 md:space-y-10">
                                <div className="border-t pt-8 md:pt-10">
                                    <h2 className="text-xl md:text-2xl font-bold mb-6">评论 ({comments.length})</h2>
                                    <CommentForm onSubmit={addComment} />
                                </div>

                                <div className="border-t pt-8 md:pt-10">
                                    <CommentList
                                        comments={comments}
                                        onReply={replyToComment}
                                    />
                                </div>
                            </div>
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