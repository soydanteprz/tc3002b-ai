import java.io.*;
import java.util.*;
import static java.lang.Math.*;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Codeforces {
    public static void main(String[] args) throws IOException {
        // InputStream inputStream = new FileInputStream("input.txt");
        // OutputStream outputStream = new FileOutputStream("output.txt");
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solver = new TaskA();
        solver.solve(in.nextInt(), in, out);
        out.close();
    }
    static class TaskA {
        long mod = (long)(1000000007);
        long fact[];
        int depth[];
        int parentTable[][];
        int degree[];
        ArrayList<Integer> leaves;
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        int diam = 0;

        public void solve(int testNumber, InputReader in, PrintWriter out) throws IOException {
            while(testNumber-->0){
                int n = in.nextInt();
                int a = in.nextInt();
                int b = in.nextInt();
                int na = in.nextInt();
                int nb = in.nextInt();

                ArrayList<ArrayList<Integer>> g = new ArrayList<>();
                for(int i=0;i<=n;i++)
                    g.add(new ArrayList<>());
                for(int i=1;i<n;i++){
                    int u = in.nextInt();
                    int v = in.nextInt();
                    g.get(u).add(v);
                    g.get(v).add(u);
                }

                if(2*na>=nb){
                    out.println("Alice");
                    continue;
                }

                // parentTable = new int[n+1][31];
                depth = new int[n+1];
                diam = 0;
                dfs(g , a , 0);

                int distance = depth[b];
                // out.println(distance);

                if(distance <= na){
                    out.println("Alice");
                    continue;
                }

                if(2*na >= min(nb , diam))
                    out.println("Alice");
                else
                    out.println("Bob");

                // max = Integer.MIN_VALUE;
                // int diameter = diameter(g , 1 , 0);
                // int height = 0;
                // for(int i=1;i<=n;i++)
                //     height = max(height , depth[i]);

                // // out.println("diameter" + diameter);
                // // out.println("height" + height);
                // // out.println("max" + max);

                // diameter = max(diameter-1 , height-1);
                // // diameter = max(diameter , max-1);

                // nb = min(nb , diameter);

                // na = min(na , diameter);


                // if(na*2 >= nb)
                //     out.println("Alice");
                // else
                //     out.println("Bob");


            }
        }

 
        int dfs(ArrayList<ArrayList<Integer>> a , int x, int p) {
            int len = 0;
            for(int y : a.get(x)) {
                if(y != p) {
                    depth[y] = depth[x] + 1;
                    int cur = 1 + dfs(a , y, x);
                    diam = max(diam, cur + len);
                    len = max(len, cur);
                }
            }
            return len;
        }

        

        // public void dfs(ArrayList<ArrayList<Integer>> a , int index , int parent){
        //     parentTable[index][0] = parent;
        //     for(int i=1;i<31;i++)
        //         parentTable[index][i] = parentTable[parentTable[parent][i-1]][i-1];

        //     depth[index] = 1+depth[parent];

        //     int l = a.get(index).size();
        //     for(int i=0;i<l;i++){
        //         if(a.get(index).get(i) == parent)
        //             continue;
        //         dfs(a , a.get(index).get(i) , index);
        //     }

        // }

        

        

        


        
        

        

        

        

        

        // for the min max problems
        
        

        // for printing purposes
        
        
        
        


        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    }

    
}
