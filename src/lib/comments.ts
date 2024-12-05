import fs from 'fs/promises'
import path from 'path'
import yaml from 'js-yaml'
import { Comment, CommentFormData } from '@/types/comment'
import { nanoid } from 'nanoid'

const commentsDirectory = path.join(process.cwd(), 'data/comments')

async function ensureDirectory() {
    try {
        await fs.access(commentsDirectory)
    } catch {
        await fs.mkdir(commentsDirectory, { recursive: true })
    }
}

export async function getComments(postSlug: string): Promise<Comment[]> {
    await ensureDirectory()
    try {
        const filePath = path.join(commentsDirectory, `${postSlug}.yaml`)
        const fileContents = await fs.readFile(filePath, 'utf8')
        return yaml.load(fileContents) as Comment[] || []
    } catch {
        return []
    }
}

export async function addComment(
    postSlug: string,
    data: CommentFormData,
    parentId?: string
): Promise<void> {
    const comments = await getComments(postSlug)
    const newComment: Comment = {
        id: nanoid(),
        author: data.author,
        email: data.email,
        content: data.content,
        createdAt: new Date().toISOString(),
    }

    if (parentId) {
        const parentComment = findComment(comments, parentId)
        if (parentComment) {
            parentComment.replies = parentComment.replies || []
            parentComment.replies.push(newComment)
        }
    } else {
        comments.push(newComment)
    }

    const filePath = path.join(commentsDirectory, `${postSlug}.yaml`)
    await fs.mkdir(commentsDirectory, { recursive: true })
    await fs.writeFile(filePath, yaml.dump(comments))
}

function findComment(comments: Comment[], id: string): Comment | null {
    for (const comment of comments) {
        if (comment.id === id) return comment
        if (comment.replies) {
            const found = findComment(comment.replies, id)
            if (found) return found
        }
    }
    return null
} 