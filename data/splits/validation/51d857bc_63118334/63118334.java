import java.util.*;
import java.io.*;
public class MyClass
{
    
    
    static ArrayList<Integer> edge[];
    public static void main(String args[])
    {
        FastReader fs=new FastReader();
        PrintWriter pw=new PrintWriter(System.out);
        int tc=fs.nextInt();
        while(tc-->0)
        {
            int n=fs.nextInt();
            int a=fs.nextInt();
            int b=fs.nextInt();
            int da=fs.nextInt();
            int db=fs.nextInt();
            edge=new ArrayList[n+1];
            for(int i=1;i<=n;i++)
            edge[i]=new ArrayList<>();
            for(int i=1;i<n;i++)
            {
                int u=fs.nextInt();
                int v=fs.nextInt();
                edge[u].add(v);
                edge[v].add(u);
            }
            int dist[]=new int[n+1];
            Arrays.fill(dist,-1);
            dist[a]=0;
            Queue<Integer> queue=new LinkedList<>();
            queue.add(a);
            while(!queue.isEmpty())
            {
                int node=queue.poll();
                for(int v:edge[node])
                {
                    if(dist[v]==-1)
                    {
                        dist[v]=dist[node]+1;
                        queue.add(v);
                    }
                }
            }
            if(dist[b]<=da)
            {
                pw.println("Alice");
                continue;
            }
            int mx=0,mxvert=1;
            for(int i=1;i<=n;i++)
            {
                if(dist[i]>mx)
                {
                    mx=dist[i];
                    mxvert=i;
                }
            }
            Arrays.fill(dist,-1);
            dist[mxvert]=0;
            queue.add(mxvert);
            while(!queue.isEmpty())
            {
                int node=queue.poll();
                for(int v:edge[node])
                {
                    if(dist[v]==-1)
                    {
                        dist[v]=dist[node]+1;
                        queue.add(v);
                    }
                }
            }
            for(int i=1;i<=n;i++)
            mx=Math.max(mx,dist[i]);
            db=Math.min(db,mx);
            if(db>2*da)
            pw.println("Bob");
            else
            pw.println("Alice");
        }
        pw.flush();
        pw.close();
    }
}
