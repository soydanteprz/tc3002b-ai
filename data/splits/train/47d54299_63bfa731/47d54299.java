import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class a {

   static int depth[];
   static ArrayList<Integer> tree[];
   static int diameter=0;

    public static void main(String[] args) {
        FastReader input=new FastReader();
        int t=input.nextInt();
        while (t-->0){
            diameter=0;
            int n=input.nextInt();
            int a=input.nextInt();
            int b=input.nextInt();
            int da=input.nextInt();
            int db=input.nextInt();
            depth=new int[n+1];
            tree=new ArrayList[n+1];
            for (int i = 0; i <=n ; i++) {
                tree[i]=new ArrayList<>();
            }
            for (int i = 0; i <n-1 ; i++) {
                int x=input.nextInt();
                int y=input.nextInt();
                tree[x].add(y);
                tree[y].add(x);
            }
            dfs(a,0);
//            System.out.println(depth[b]);
            if (2 * da >= Math.min(diameter, db) ||depth[b]<=da){
                System.out.println("Alice");
            }else {
                System.out.println("Bob");
            }



       }

    }

    private static int dfs(int root, int p) {
        int len=0;

        for(int child:tree[root]){
            if (child!=p){
                depth[child]=depth[root]+1;

                int curr=1+dfs(child,root);
                diameter=Math.max(diameter,curr+len);
                len=Math.max(len,curr);

            }
        }
        return len;

    }
    

}
