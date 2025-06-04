import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;


public class Main {

    public static void main(String[] args) {

        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader sc = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Solver solver = new Solver();
		int t = sc.nextInt();
//        int t = 1;
        while (t-- != 0) {
            solver.solve(sc, out);
        }
        out.close();

    }

    static class Solver {
        ArrayList<Integer>[] gp;
        int theirdist = 0;
        int dist = 0;
        boolean[] vis;
        int first;
        int sec;
        public void solve(InputReader sc, PrintWriter out) {
            int n = sc.nextInt();
            int a = sc.nextInt()-1;
            int b = sc.nextInt()-1;
            int da = sc.nextInt();
            int db = sc.nextInt();dist=0;first=0;sec=0;
            vis = new boolean[n];
            theirdist = 0;
            gp = new ArrayList[n];
            for(int i = 0; i < n; i++) {
                gp[i] = new ArrayList<>();
            }
            for(int i = 0; i < n-1; i++) {
                int s = sc.nextInt()-1;
                int d = sc.nextInt()-1;
                gp[s].add(d);
                gp[d].add(s);
            }
            dfsfirst(0,0);
            Arrays.fill(vis,false);
            dist = 0;
            dfssec(first,0);
            int dia = dist;
            Arrays.fill(vis,false);
            dfstheir(a,b,0);
            if(theirdist<=da) {
                out.println("Alice");
                return;
            }
            if(da>=(dia+1)/2) {
                out.println("Alice");
                return;
            }
            if(db<(da*2)+1) {
                out.println("Alice");
                return;
            }
            out.println("Bob");
        }

        private void dfstheir(int a, int b, int i) {
            vis[a] = true;
            if(a==b) {
                theirdist = i;
                //return;
            }
            for(int v : gp[a]) {
                if(!vis[v]) {
                    dfstheir(v,b,i+1);
                }
            }
        }

        private void dfssec(int first, int i1) {
            vis[first] = true;
            for(int v : gp[first]) {
                if(!vis[v]) {
                    dfssec(v,i1+1);
                }
            }
            if(i1>=dist) {
                dist = i1;
                sec = first;
            }
        }

        private void dfsfirst(int i, int i1) {
            vis[i] = true;
            for(int v : gp[i]) {
                if(!vis[v]) {
                    dfsfirst(v,i1+1);
                }
            }
            if(i1>=dist) {
                dist = i1;
                first = i;
            }
        }
    }

    

    

    

    

    


}
