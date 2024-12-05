/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        remotePatterns: [
            {
                protocol: 'https',
                hostname: '**',
            },
            {
                protocol: 'http',
                hostname: '**',
            }
        ],
    },
    eslint: {
        // 在生产构建时忽略 ESLint 错误
        ignoreDuringBuilds: true,
    },
}

module.exports = nextConfig 