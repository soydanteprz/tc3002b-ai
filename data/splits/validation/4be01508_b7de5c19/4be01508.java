import java.util.*;
import java.io.*;

public class q3{
    static FastScanner fs = new FastScanner();
    static PrintWriter pw = new PrintWriter(System.out);
    static List<List<Edge>> list;
    public static void main(String[] args){
        int tt = fs.nextInt();
        for (int t=0;t<tt;t++){
            solve();
        }
        pw.close();
    }

    static void solve(){
        int n = fs.nextInt();
        list = new ArrayList<>();
        for (int i=0;i<n;i++) list.add(new ArrayList<>());
        
        for (int i=0;i<n-1;i++){
            int from = fs.nextInt()-1, to = fs.nextInt()-1;
            list.get(from).add(new Edge(to, i));
            list.get(to).add(new Edge(from, i));
        }

        int start = -1;
        for (int i=0;i<n;i++){
            if (list.get(i).size() > 2){
                pw.println(-1);
                return;
            }
            else if (list.get(i).size() == 1){
                start = i;
            }
        }   

        int[] ans = new int[n-1];
        int prev = -1;
        int cur = start;
        int curWeight = 2;
        while (true){
            Edge now = list.get(cur).get(0);
            if (now.node == prev){
                if (list.get(cur).size() == 1){
                    break;
                }
                now = list.get(cur).get(1);
            }

            ans[now.index] = 5 - curWeight;
            curWeight = ans[now.index];
            prev = cur;
            cur = now.node;
        }

        for (int i : ans){
            pw.printf("%d ", i);
        }
        pw.println();
    }

    static class Edge {
        int node;
        int index;

        public Edge(int node, int index){
            this.node = node;
            this.index = index;
        }
    }

//    ----------input function----------

    

    
}
