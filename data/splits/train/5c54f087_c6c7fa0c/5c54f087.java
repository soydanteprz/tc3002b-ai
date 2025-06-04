import java.util.*;
import java.lang.*;
import java.io.*;

public class Main
{
	PrintWriter out = new PrintWriter(System.out);
	BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
    StringTokenizer tok = new StringTokenizer("");
    String next() throws IOException {
        if (!tok.hasMoreTokens()) { tok = new StringTokenizer(in.readLine()); }
        return tok.nextToken();
    }
    int ni() throws IOException { return Integer.parseInt(next()); }
    
    
    int n,a,b,da,db,dist,vert;
    ArrayList<Integer>A[];
    
    void solve() throws IOException {
        for (int tc=ni();tc>0;tc--) {
            n=ni(); a=ni(); b=ni(); da=ni(); db=ni();
            A=new ArrayList[n+1];
            for (int i=1;i<=n;i++) A[i]=new ArrayList();
            for (int i=1;i<n;i++) {
                int u=ni(),v=ni();
                A[u].add(v);
                A[v].add(u);
            }
            
            dist=0;
            dfs1(a,0,0);
            if (dist<=da || db<=2*da) {
                out.println("Alice");
                continue;
            }
            
            dist=0;
            vert=0;
            dfs2(1,0,0);
            dist=0;
            dfs2(vert,0,0);
            
            if (dist<=2*da) out.println("Alice");
            else out.println("Bob");
        }
        out.flush();
    }
    
    void dfs2(int u,int p,int d) {
        if (d>dist) {
            dist=d;
            vert=u;
        }
        
        for (Integer v:A[u]) {
            if (v==p) continue;
            dfs2(v,u,d+1);
        }
    }
    
    void dfs1(int u,int p,int d) {
        if (u==b) dist=d;
        for (Integer v:A[u]) {
            if (v==p) continue;
            dfs1(v,u,d+1);
        }
    }
    
    public static void main(String[] args) throws IOException {
        new Main().solve();
    }
}
