import { addComment } from "@/lib/comments"
import { NextRequest, NextResponse } from "next/server"
import { CommentFormData } from "@/types/comment"

export async function POST(
    request: NextRequest,
    { params }: { params: { slug: string } }
) {
    try {
        const { slug } = await Promise.resolve(params)

        if (!slug) {
            return NextResponse.json(
                { error: "Missing slug parameter" },
                { status: 400 }
            )
        }

        const body = await request.json()
        const data = body.data as CommentFormData
        const parentId = body.parentId as string | undefined

        await addComment(slug, data, parentId)

        return NextResponse.json({ success: true })
    } catch (error) {
        console.error('Error adding comment:', error)
        return NextResponse.json(
            { error: "Failed to add comment" },
            { status: 500 }
        )
    }
} 