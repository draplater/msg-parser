
/*
    对图在1ec限制下做dp，返回选取的边集
    建议放于 graphcjj1_oepcross_depparser.h 同路径下
    调用方式
        include 文件之后
        graphhs_oepcross::parse( num_point, Score, m_vecTrainArcs );
        其中
            int num_point; 为图中点的个数
            tscore Score[MAX_SENTENCE_SIZE][MAX_SENTENCE_SIZE][2]; 为边的分数
            vector<Arc> m_vecTrainArcs; 用于保存被选择的边
*/

#include <cstdio>
#include <vector>

/*
#include "graphcjj1_oepcross_depparser.h"
#include "common/token/word.h"
#include "common/token/pos.h"
*/

#define tscore double
#define MAX_SENTENCE_SIZE 256
#define Arc BiGram<long>

template<class T>
class BiGram
{
public:
    T first, second;
    BiGram(T x,T y): first(x), second(y) {}
};


using namespace std;

namespace graphhs_oepcross
{
    using namespace std;
    const int MAX_DP_FROM = 3;  // dp 最大转移来源数
    const int MAX_DP_EDGE = 3;  // dp 最大边加入数

    const int TYPE_INT = 0;
    const int TYPE_N   = 1;
    const int TYPE_L   = 2;
    const int TYPE_R   = 3;
    const int TYPE_LR  = 4;
    const int TYPE_MAX = 4;

    //vector< vector<tscore> > s; // 边的分数
    //const tscore (*edge_score)[MAX_SENTENCE_SIZE][2];
    class StateItem 
    {
    public:
        //int type;
        //int left, right, cross;
        tscore score;
        StateItem *from[MAX_DP_FROM];
        int add_edge[MAX_DP_EDGE][2];
        
        void Reset()
        {
            score = 0;
            return;
        }
        void Update( const tscore edge_score[][MAX_SENTENCE_SIZE][2], bool set, vector<StateItem*> f, vector<int> edge )
        {
            tscore sum = 0;
            int e[MAX_DP_EDGE][2]={0};
            for( int i=0; i<f.size(); i++ )
                sum += f[i]->score;
            for( int i=0, x,y; (i<<1)<edge.size(); i++ )
            {
                x = edge[i<<1];
                y = edge[(i<<1)+1];
                if( edge_score[y][x][0] > edge_score[x][y][0] )
                    swap(x,y);
                if( edge_score[x][y][0]<=0 )
                    continue;
                e[i][0] = x;
                e[i][1] = y;
                sum += edge_score[x][y][0];
            }
            if( set || sum>score )
            {
                score = sum;
                for( int i=0; i<MAX_DP_FROM; i++ )
                    if( i<f.size() )
                        from[i] = f[i];
                    else
                        from[i] = 0;
                for( int i=0; i<MAX_DP_EDGE; i++ )
                    for( int k=0; k<2; k++ )
                        add_edge[i][k] = e[i][k];
            }
            return;
        }
    };
    typedef vector< vector< vector< vector<StateItem> > > > dp_stru;
    
    //int num_point; // 点的个数
    //vector< vector< vector< vector<StateItem> > > > dp; // dp 状态存储
    
    dp_stru* Init(int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2]) // 建立dp空间
    {
        // 建立dp空间
        dp_stru *dp_p = new dp_stru;
        dp_stru &dp = *dp_p;
        
        dp.resize( TYPE_MAX+1 );
        for( int t=0; t<=TYPE_MAX; t++ )
        {
            dp[t].resize( num_point+1 );
            for( int l=1; l<=num_point; l++ )
            {
                dp[t][l].resize( num_point+1 );
                for( int r=1; r<=num_point; r++ )
                    dp[t][l][r].resize( num_point+1 );
            }
        }
        return dp_p;
    }
    void Clear_All(int num_point, dp_stru &dp) // 释放所有申请的空间
    {
        for( int t=0; t<=TYPE_MAX; t++ )
        {
            for( int l=1; l<=num_point; l++ )
            {
                for( int r=1; r<=num_point; r++ )
                    dp[t][l][r].clear();
                dp[t][l].clear();
            }
            dp[t].clear();
        }
        dp.clear();
        delete &dp;
        return;
    }


    void Dp_LRN( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x);
    tscore Dynamic_Programming( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp) // 对 Int 类型进行动归，并返回整个图的最优值
    {
        // 初始化
        for( int i=1; i<num_point; i++ )
        {
            dp[TYPE_INT][i][i+1][0].Update( edge_score, true, {}, { i,i+1 } );
            for( int x=1; x<=num_point; x++ )
                for( int t=TYPE_N; t<=TYPE_LR; t++ )
                    dp[t][i][i+1][x].Update( edge_score, true, {}, { i,i+1 } );
        }
        
        // Int
        for( int dist=2; dist<num_point; dist++ )
            for( int i=1, j; (j=i+dist) <= num_point; i++ )
            {
                dp[TYPE_INT][i][j][0].Reset();
                for( int k=i+1; k<j; k++ )
                {
                    dp[TYPE_INT][i][j][0].Update( edge_score, false, 
                        { &dp[TYPE_LR][i][k][j], &dp[TYPE_INT][k][j][0] }, { i,j } );                
                    for( int l=k+1; l<j; l++ )
                    {
                        dp[TYPE_INT][i][j][0].Update( edge_score, false,
                            { &dp[TYPE_R][i][k][l], &dp[TYPE_INT][k][l][0], &dp[TYPE_L][l][j][k] },  
                            { i,l, k,j, i,j } );
                        dp[TYPE_INT][i][j][0].Update( edge_score, false,
                            { &dp[TYPE_LR][i][k][l], &dp[TYPE_INT][k][l][0], &dp[TYPE_INT][l][j][0] },
                            { i,l, i,j } );
                    }
                    for( int l=i+1; l<k; l++ )
                    {
                        dp[TYPE_INT][i][j][0].Update( edge_score, false,
                            { &dp[TYPE_INT][i][l][0], &dp[TYPE_L][l][k][i], &dp[TYPE_N][k][j][l] },
                            { i,k, l,j, i,j } );
                        dp[TYPE_INT][i][j][0].Update( edge_score, false,
                            { &dp[TYPE_R][i][l][k], &dp[TYPE_INT][l][k][0], &dp[TYPE_L][k][j][l] },
                            { i,k, l,j, i,j } );
                    }
                }
                
                for( int x=1; x<i; x++ )
                    Dp_LRN( num_point, edge_score, dp, i, j, x );
                for( int x=j+1; x<=num_point; x++ )
                    Dp_LRN( num_point, edge_score, dp, i, j, x );
                
            }
        return dp[TYPE_INT][1][num_point][0].score;
    }
    void Dp_LRN( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x) // 对 Int 之外的类型动归
    {
        // N
        dp[TYPE_N][i][j][x].Update( edge_score, true, { &dp[TYPE_INT][i][j][0] }, {} );
        for( int k=i+1; k<j; k++ )
            dp[TYPE_N][i][j][x].Update( edge_score, false, 
                { &dp[TYPE_N][i][k][x], &dp[TYPE_INT][k][j][0] }, { x,k } );
        
        // L
        dp[TYPE_L][i][j][x].Update( edge_score, true, { &dp[TYPE_INT][i][j][0] }, {} );
        for( int k=i+1; k<j; k++ )
        {
            dp[TYPE_L][i][j][x].Update( edge_score, false, 
                { &dp[TYPE_L][i][k][x], &dp[TYPE_N][k][j][i] }, { x,k, i,j } );
            dp[TYPE_L][i][j][x].Update( edge_score, false, 
                { &dp[TYPE_INT][i][k][0], &dp[TYPE_L][k][j][i] }, { x,k, i,j } );
        }
        
        // R
        dp[TYPE_R][i][j][x].Update( edge_score, true, { &dp[TYPE_INT][i][j][0] }, {} );
        for( int k=i+1; k<j; k++ )
        {
            dp[TYPE_R][i][j][x].Update( edge_score, false, 
                { &dp[TYPE_N][i][k][j], &dp[TYPE_R][k][j][x] }, { x,k, i,j } );
            dp[TYPE_R][i][j][x].Update( edge_score, false, 
                { &dp[TYPE_R][i][k][j], &dp[TYPE_INT][k][j][0] }, { x,k, i,j } );
        }

        // LR
        dp[TYPE_LR][i][j][x].Update( edge_score, true, { &dp[TYPE_L][i][j][x] }, {} );
        dp[TYPE_LR][i][j][x].Update( edge_score, false, { &dp[TYPE_R][i][j][x] }, {} );
        for( int k=i+1; k<j; k++ )
            dp[TYPE_LR][i][j][x].Update( edge_score, false,
                { &dp[TYPE_L][i][k][x], &dp[TYPE_R][k][j][x] }, { x,k, i,j } );
        return;
    }


    void Dfs_Arc( StateItem *p, vector< Arc > &r ) // Dfs 遍历所有边
    {
        for( int i=0; i<MAX_DP_FROM; i++ )
            if( p->from[i]!=NULL )
                Dfs_Arc( p->from[i], r );
        for( int i=0; i<MAX_DP_EDGE; i++ )
            if( p->add_edge[i][0] )
                r.push_back( BiGram<long>( p->add_edge[i][0], p->add_edge[i][1] ) );
        return;
    }

    void parse(int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], vector< Arc > &result) // 主函数
    {
        dp_stru *dp_p = Init( num_point, edge_score );
        
        Dynamic_Programming( num_point, edge_score, *dp_p);
        
        result.clear();
        Dfs_Arc( &(*dp_p)[TYPE_INT][1][num_point][0], result );
        Clear_All( num_point, *dp_p );
        return;
    }

}

extern "C" {
    extern int parse(int num_point, tscore Score[][MAX_SENTENCE_SIZE][2], int arcs[][2]) {
        vector<Arc> result;
        graphhs_oepcross::parse(num_point, Score, result);
        for(int i=0; i<result.size(); i++) {
            arcs[i][0] = result[i].first;
            arcs[i][1] = result[i].second;
        }
        return result.size();
    }
}
