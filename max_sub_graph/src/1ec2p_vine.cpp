
/*
    对图在1ec2p以及最大边长度限制下做dp，返回选取的边集
    调用方式
        include 文件之后
        graphhs_oepcross::parse( max_arc_len, num_point, Score, m_vecTrainArcs );
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

    bool Short(int x, int y, int max_arc_len)
    {
        int v=y-x;
        if( v>0 )
            return v<=max_arc_len;
        else
            return -v<=max_arc_len;
    }
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
        void Update( int max_arc_len, const tscore edge_score[][MAX_SENTENCE_SIZE][2], bool set, vector<StateItem*> f, vector<int> edge )
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
                if( edge_score[x][y][0]<=0 || !Short(x,y,max_arc_len) )
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
    
    //int num_point; // 点的个数
    //vector< vector< vector< vector<StateItem> > > > dp; // dp 状态存储
    class DP_ARRAY
    {
    private:
        vector< vector< vector< vector<StateItem> > > > data;
        const int num_point;
        const int max_arc_len;
    public:
        DP_ARRAY(const int len, const int n_point): num_point(n_point), max_arc_len(len)
        {
            // 建立dp空间
            data.resize( TYPE_MAX+1 );
            for( int t=0; t<=TYPE_MAX; t++ )
            {
                data[t].resize( num_point+1 );
                for( int l=1; l<=num_point; l++ )
                {
                    /*
                    data[t][l].resize( max_arc_len+2 );
                    for( int r=1; r<=max_arc_len+1; r++ )
                        data[t][l][r].resize( max_arc_len*3+2 );
                    */
                    data[t][l].resize( min( num_point+1, max_arc_len+2 ) );
                        for( int r=1; r<min( num_point+1, max_arc_len+2 ); r++ )
                        data[t][l][r].resize( min( num_point+max_arc_len+1, max_arc_len*3+2 ) );
                }
            }
            return;
        }
        StateItem & operator()(int t, int i, int j, int x)
        {
            /*
            if( !( t>=0 && t<=TYPE_MAX 
                   && i>0 && i<=num_point
                   && j>i && j<=num_point && (j-i<=max_arc_len || j==num_point)
                   && x>=0 && x<=num_point
                   && ( 
                       ( t!=TYPE_INT && x-i+max_arc_len>=0 && x-i+max_arc_len<=max_arc_len*3 )
                       ||
                       ( (t==TYPE_INT) && x==0 ) ) ) )
            {
                printf("ASSERT ERROR: %d %d %d %d\n", t, i, j, x);
                printf("arc_len: %d,   num_point:%d\n", max_arc_len, num_point);
            }
            assert( t>=0 && t<=TYPE_MAX 
                   && i>0 && i<=num_point
                   && j>i && j<=num_point && (j-i<=max_arc_len || j==num_point)
                   && x>=0 && x<=num_point
                   && ( 
                       ( t!=TYPE_INT && x-i+max_arc_len>=0 && x-i+max_arc_len<=max_arc_len*3 )
                       ||
                       ( (t==TYPE_INT) && x==0 ) ) );
            */
            return data[t][i][ min( max_arc_len+1, j-i ) ][ (!!t)*(x-i+max_arc_len+1)  ];
        }
    };
    //typedef vector< vector< vector< vector<StateItem> > > > dp_stru;
    typedef DP_ARRAY dp_stru;
    void Dp_Int( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int max_arc_len)
    {
        //printf("\tInt: %d %d\n", i,j);
        dp(TYPE_INT, i, j, 0).Reset();
        for( int k=i+1; k<j && Short(i,k,max_arc_len); k++ )
        {
            if( Short(k,j,max_arc_len) )
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false, 
                    { &dp(TYPE_LR, i, k, j), &dp(TYPE_INT, k, j, 0) }, { i,j } );                
            else
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false, 
                    { &dp(TYPE_INT, i, k, 0), &dp(TYPE_INT, k, j, 0) }, { i,j } );                
            for( int l=k+1; l<j && Short(k,l,max_arc_len); l++ )
            {
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false,
                    { &dp(TYPE_R, i, k, l), &dp(TYPE_INT, k, l, 0), &dp(TYPE_L, l, j, k) },  
                    { i,l, k,j, i,j } );
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false,
                    { &dp(TYPE_LR, i, k, l), &dp(TYPE_INT, k, l, 0), &dp(TYPE_INT, l, j, 0) },
                    { i,l, i,j } );
            }
            for( int l=i+1; l<k; l++ )
            {
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false,
                    { &dp(TYPE_INT, i, l, 0), &dp(TYPE_L, l, k, i), &dp(TYPE_N, k, j, l) },
                    { i,k, l,j, i,j } );
                dp(TYPE_INT, i, j, 0).Update( max_arc_len, edge_score, false,
                    { &dp(TYPE_R, i, l, k), &dp(TYPE_INT, l, k, 0), &dp(TYPE_L, k, j, l) },
                    { i,k, l,j, i,j } );
            }
        }
        return;
    }
    void Dp_N( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x, int max_arc_len)
    {
        //printf("\tN: %d %d %d\n", i,j,x);
        dp(TYPE_N, i, j, x).Update( max_arc_len, edge_score, true, { &dp(TYPE_INT, i, j, 0) }, {} );
        for( int k=i+1; k<j && (x>j || Short(x,k,max_arc_len)); k++ )
            dp(TYPE_N, i, j, x).Update( max_arc_len, edge_score, false, 
                { &dp(TYPE_N, i, k, x), &dp(TYPE_INT, k, j, 0) }, { x,k } );
        return;
    }
    void Dp_L( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x, int max_arc_len)
    {
        //printf("\tL: %d %d %d\n", i,j,x);
        dp(TYPE_L, i, j, x).Update( max_arc_len, edge_score, true, { &dp(TYPE_INT, i, j, 0) }, {} );
        for( int k=i+1; k<j && (x>j || Short(x,k,max_arc_len)); k++ )
        {
            dp(TYPE_L, i, j, x).Update( max_arc_len, edge_score, false, 
                { &dp(TYPE_L, i, k, x), &dp(TYPE_N, k, j, i) }, { x,k, i,j } );
            dp(TYPE_L, i, j, x).Update( max_arc_len, edge_score, false, 
                { &dp(TYPE_INT, i, k, 0), &dp(TYPE_L, k, j, i) }, { x,k, i,j } );
        }
        return;
    }
    void Dp_R( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x, int max_arc_len)
    {
        //printf("\tR: %d %d %d\n", i,j,x);
        dp(TYPE_R, i, j, x).Update( max_arc_len, edge_score, true, { &dp(TYPE_INT, i, j, 0) }, {} );
        for( int k=i+1; k<j; k++ )
        {
            dp(TYPE_R, i, j, x).Update( max_arc_len, edge_score, false, 
                { &dp(TYPE_N, i, k, j), &dp(TYPE_R, k, j, x) }, { x,k, i,j } );
            dp(TYPE_R, i, j, x).Update( max_arc_len, edge_score, false, 
                { &dp(TYPE_R, i, k, j), &dp(TYPE_INT, k, j, 0) }, { x,k, i,j } );
        }
        return;
    }
    void Dp_LR( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int i, int j, int x, int max_arc_len)
    {
        //printf("\tLR: %d %d %d\n", i,j,x);
        dp(TYPE_LR, i, j, x).Update( max_arc_len, edge_score, true, { &dp(TYPE_L, i, j, x) }, {} );
        dp(TYPE_LR, i, j, x).Update( max_arc_len, edge_score, false, { &dp(TYPE_R, i, j, x) }, {} );
        for( int k=i+1; k<j; k++ )
            dp(TYPE_LR, i, j, x).Update( max_arc_len, edge_score, false,
                { &dp(TYPE_L, i, k, x), &dp(TYPE_R, k, j, x) }, { x,k, i,j } );
        return;
    }

    tscore Dynamic_Programming( int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], dp_stru &dp, int max_arc_len) // 进行动归，并返回整个图的最优值
    {
        // 初始化
        for( int i=1; i<num_point; i++ )
        {
            dp(TYPE_INT, i, i+1, 0).Update( max_arc_len, edge_score, true, {}, { i,i+1 } );
            for( int x=1; x<=num_point; x++ )
                if( Short(x,i,max_arc_len) || Short(x,i+1,max_arc_len) )
                    for( int t=TYPE_N; t<=TYPE_LR; t++ )
                        dp(t, i, i+1, x).Update( max_arc_len, edge_score, true, {}, { i,i+1 } );
        }
        
        for( int dist=2; dist<num_point && dist<=max_arc_len; dist++ )
            for( int i=1, j; (j=i+dist) <= num_point; i++ )
            {
                Dp_Int( num_point, edge_score, dp, i, j, max_arc_len );
                for( int x=max(1,i-max_arc_len); x<i; x++ )
                {
                    Dp_N( num_point, edge_score, dp, i, j, x, max_arc_len );
                    Dp_L( num_point, edge_score, dp, i, j, x, max_arc_len );
                }
                for( int x=j+1; x<=num_point && Short(x,j,max_arc_len); x++ )
                {
                    Dp_N( num_point, edge_score, dp, i, j, x, max_arc_len );
                    Dp_L( num_point, edge_score, dp, i, j, x, max_arc_len );
                    Dp_R( num_point, edge_score, dp, i, j, x, max_arc_len );
                    Dp_LR( num_point, edge_score, dp, i, j, x, max_arc_len );
                }
                
            }
        for( int dist=max_arc_len+1; dist<num_point; dist++ )
        {
            int i=num_point-dist, j=num_point;
            Dp_Int( num_point, edge_score, dp, i, j, max_arc_len );
            for( int x=max(1,i-max_arc_len); x<i; x++ )
            {
                Dp_N( num_point, edge_score, dp, i, j, x, max_arc_len );
                Dp_L( num_point, edge_score, dp, i, j, x, max_arc_len );
            }
        }
        if( num_point<2 )
            return 0;
        else
            return dp(TYPE_INT, 1, num_point, 0).score;
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

    void parse(int max_arc_len, int num_point, const tscore edge_score[][MAX_SENTENCE_SIZE][2], vector< Arc > &result) // 主函数
    {
        dp_stru dp( max_arc_len, num_point );
        Dynamic_Programming( num_point, edge_score, dp, max_arc_len);
        result.clear();
        if( num_point>1 )
            Dfs_Arc( &dp(TYPE_INT, 1, num_point, 0), result );
        return;
    }

}

extern "C" {
    extern int parse_vine(int max_arc_len, int num_point, tscore Score[][MAX_SENTENCE_SIZE][2], int arcs[][2]) {
        std::vector<Arc> result;
        graphhs_oepcross::parse(max_arc_len, num_point, Score, result);
        for(int i=0; i<result.size(); i++) {
            arcs[i][0] = result[i].first;
            arcs[i][1] = result[i].second;
        }
        return result.size();
    }
}