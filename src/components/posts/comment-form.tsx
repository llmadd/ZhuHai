'use client'

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { useState, useRef } from "react"
import { CommentFormData } from "@/types/comment"

interface CommentFormProps {
    onSubmit: (data: CommentFormData) => Promise<void>
    replyTo?: string
}

export function CommentForm({ onSubmit, replyTo }: CommentFormProps) {
    const [isSubmitting, setIsSubmitting] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const formRef = useRef<HTMLFormElement>(null)

    async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault()
        setIsSubmitting(true)
        setError(null)

        const formData = new FormData(event.currentTarget)
        const data: CommentFormData = {
            author: formData.get('author') as string,
            email: formData.get('email') as string,
            content: formData.get('content') as string,
        }

        try {
            await onSubmit(data)
            formRef.current?.reset()
        } catch (err) {
            setError('提交评论失败，请稍后重试')
            console.error(err)
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <form ref={formRef} onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                    name="author"
                    placeholder="昵称"
                    required
                    minLength={2}
                />
                <Input
                    name="email"
                    type="email"
                    placeholder="邮箱"
                    required
                />
            </div>
            <Textarea
                name="content"
                placeholder={replyTo ? `回复 ${replyTo}...` : "写下你的评论..."}
                required
                minLength={2}
                className="min-h-[100px]"
            />
            <div className="flex items-center gap-4">
                <Button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "提交中..." : "发表评论"}
                </Button>
                {error && (
                    <p className="text-red-500 text-sm">{error}</p>
                )}
            </div>
        </form>
    )
} 