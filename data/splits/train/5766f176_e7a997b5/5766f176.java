import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        // write your code here
        boolean readFromLocal = true;
        //readFromLocal = false;
        String filepath = "src/input.txt";
        //FileInputStrviseam fileInputStream = new FileInputStream(filepath);
        InputReader inputReader = new InputReader(System.in);
        Solve s = new Solve();
        s.solve(inputReader);
    }
}

class Solve {

    public void solve(InputReader inputReader) {
        int t,n;
        t = inputReader.nextInt();
        while (t>0) {
            t--;
            n = inputReader.nextInt();
            Graph g = new Graph(n);
            g.res = new int[n-1];
            for(int i=1;i<n;i++){
                g.addEdge(inputReader.nextInt(), inputReader.nextInt(),i-1,true);
            }
            if (g.hasDegreeMoreThanTwo()){
                System.out.println(-1);
            }else {
                int minDegree = 2,node = 1;
                for(int i=1;i<=n;i++){
                    if (g.adj[i].size()<minDegree){
                        node = i;
                        minDegree = g.adj[i].size();
                    }
                }
                g.dfs(node,-1,2);
                for(int i=0;i<n-1;i++){
                    System.out.print(g.res[i] + " ");
                }
                System.out.println();
            }
        }
    }
}








class Graph {
    public ArrayList<int[]>[] adj;
    int size;
    boolean[] vis;
    public int [] res;
    Graph(int n){
        this.size = n;
        this.adj = new ArrayList[n+1];
        this.vis = new boolean[n+1];
        for (int i = 0; i <=n; i++) {
            adj[i] = new ArrayList<>();
        }
    }
    public void addEdge(int a, int b, int edge,boolean biDirectional){
        adj[a].add(new int[]{b, edge});
        if (biDirectional) {
            adj[b].add(new int[]{a, edge});
        }
    }

    public void dfs(int node,int parent,int prime){
        for (int[] nodePair: adj[node]) {
            if (nodePair[0]!=parent){
                res[nodePair[1]] = prime;
                dfs(nodePair[0],node,prime^1);
            }
        }
    }
    boolean hasDegreeMoreThanTwo() {
        for (int i = 0; i <= size; i++) {
            if (adj[i].size()>2){
                return true;
            }
        }
        return false;
    }

}
