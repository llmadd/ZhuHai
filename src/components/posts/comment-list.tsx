"use client"

import { Comment } from "@/types/comment"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { format } from "date-fns"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { CommentForm } from "./comment-form"

interface CommentListProps {
    comments: Comment[]
    onReply: (commentId: string, data: { author: string; email: string; content: string }) => Promise<void>
}

export function CommentList({ comments, onReply }: CommentListProps) {
    const [replyingTo, setReplyingTo] = useState<string | null>(null)

    return (
        <div className="space-y-6">
            {comments.map((comment) => (
                <div key={comment.id} className="space-y-4">
                    <div className="flex items-start gap-4">
                        <Avatar>
                            <AvatarImage src={comment.avatar} />
                            <AvatarFallback>{comment.author[0]}</AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                            <div className="flex items-center gap-2">
                                <span className="font-semibold">{comment.author}</span>
                                <span className="text-sm text-muted-foreground">
                                    {format(new Date(comment.createdAt), 'yyyy年MM月dd日 HH:mm')}
                                </span>
                            </div>
                            <p className="mt-1">{comment.content}</p>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="mt-2"
                                onClick={() => setReplyingTo(replyingTo === comment.id ? null : comment.id)}
                            >
                                {replyingTo === comment.id ? '取消回复' : '回复'}
                            </Button>
                        </div>
                    </div>

                    {replyingTo === comment.id && (
                        <div className="ml-12">
                            <CommentForm
                                onSubmit={async (data) => {
                                    await onReply(comment.id, data)
                                    setReplyingTo(null)
                                }}
                                replyTo={comment.author}
                            />
                        </div>
                    )}

                    {comment.replies && comment.replies.length > 0 && (
                        <div className="ml-12 space-y-4">
                            {comment.replies.map((reply) => (
                                <div key={reply.id} className="flex items-start gap-4">
                                    <Avatar>
                                        <AvatarImage src={reply.avatar} />
                                        <AvatarFallback>{reply.author[0]}</AvatarFallback>
                                    </Avatar>
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <span className="font-semibold">{reply.author}</span>
                                            <span className="text-sm text-muted-foreground">
                                                {format(new Date(reply.createdAt), 'yyyy年MM月dd日 HH:mm')}
                                            </span>
                                        </div>
                                        <p className="mt-1">{reply.content}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            ))}
        </div>
    )
} 