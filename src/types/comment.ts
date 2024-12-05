export interface Comment {
    id: string
    author: string
    email: string
    content: string
    avatar?: string
    createdAt: string
    replies?: Comment[]
}

export interface CommentFormData {
    author: string
    email: string
    content: string
} 