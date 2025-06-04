import java.io.*;
import java.util.*;
public class C
{
    private static ArrayDeque<Integer>[] edge;
    private static HashMap<String,Integer> map;

    private static String getHash(int u, int v)
    {
        if(u>v)
        {
            int tmp=u;
            u=v;
            v=tmp;
        }

        return u+" "+v;
    }
    private static void DFS(int u, int p, int[] ans, int val)
    {
        for(int v:edge[u])
        {
            if(v==p) continue;
            ans[map.get(getHash(u,v))]=val;

            DFS(v,u,ans,5-val);
            val=5-val;
        }
    }
    public static void main(String[] args) throws Exception
    {
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));

        int i,N;

        int T=Integer.parseInt(br.readLine().trim());
        StringBuilder sb=new StringBuilder();

        while (T-->0)
        {
            N=Integer.parseInt(br.readLine().trim());
            edge=new ArrayDeque[N];
            for(i=0;i<N;i++) edge[i]=new ArrayDeque<>();

            map=new HashMap<>();
            int[] ans=new int[N-1];
            int[] deg=new int[N];

            for(i=0;i<N-1;i++)
            {
                String[] s=br.readLine().trim().split(" ");
                int u=Integer.parseInt(s[0])-1;
                int v=Integer.parseInt(s[1])-1;

                edge[u].add(v); edge[v].add(u);
                deg[u]++; deg[v]++;

                map.put(getHash(u,v),i);
            }

            for(i=0;i<N;i++) if(deg[i]>2) break;
            if(i<N)
            {
                sb.append(-1).append("\n");
                continue;
            }

            DFS(0,0,ans,2);
            for(int x:ans) sb.append(x).append(" ");
            sb.append("\n");
        }
        System.out.println(sb);
    }
}
