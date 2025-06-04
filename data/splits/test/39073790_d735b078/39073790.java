import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.*;

public class DisJump {

    public static void main(String[] args) {

        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader sc = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        Solver solver = new Solver();
//		int t = sc.nextInt();
        int t = 1;
        while (t-- != 0) {
            solver.solve(sc, out);
        }
        out.close();

    }

    

    static class Solver {
        public void solve(InputReader sc, PrintWriter out) {
            int n = sc.nextInt();
            int[] h = sc.nextIntArray(n);
            ArrayList<Integer>[] g = new ArrayList[n];
            for(int i = 0; i < n; i++) g[i] = new ArrayList<>();
            for(int rot = 0; rot < 2; rot++) {
                int[] stk = new int[n];
                int at = -1;
                for(int i = 0; i < n; i++) {
                    while (at!=-1 && h[stk[at]]<h[i]) g[stk[at--]].add(i);
                    if(at!=-1) {
                        g[stk[at]].add(i);
                        if(h[stk[at]]==h[i]) at--;
                    }
                    stk[++at] = i;
                }
                for(int i = 0; i < n; i++) h[i] = -h[i];
            }
            int[] qu = new int[n];
            int[] dist = new int[n];
            Arrays.fill(dist,-1);
            int qTop = 0;
            int qEnd = 1;
            qu[0] = 0;
            dist[0] = 0;
            while (qTop<qEnd) {
                int front = qu[qTop];
                for(int v : g[front]) {
                    if(dist[v]==-1) {
                        qu[qEnd] = v;
                        dist[v] = dist[front]+1;
                        qEnd++;
                    }
                }
                qTop++;
            }
            out.println(dist[n-1]);
        }
    }

    

    

    

    

    


}
