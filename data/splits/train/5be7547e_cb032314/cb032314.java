import java.io.*;
import java.util.*;

public class C {

    static FastScanner sc = new FastScanner();
    static PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));
    static class pair{
        public pair(int x, int y) {
            this.x = x;
            this.y = y;
        }

        int x;
        int y;
        

        
    }
    static void solve(){
        int n = sc.nextInt();
        int[]degree = new int[n];
        List<Integer>[]grid = new List[n];
        for (int i = 0; i < n; i++) {
            grid[i] = new ArrayList();
        }
        List<pair>list = new ArrayList<>();
        for (int i = 0; i < n - 1; i++) {
            int x = sc.nextInt() - 1;
            int y = sc.nextInt() - 1;
            grid[x].add(y);
            grid[y].add(x);
            degree[x]++;
            degree[y]++;
            list.add(new pair(Math.min(x,y),Math.max(x,y)));
        }
        int begin = 0;
        for(int i = 0;i < degree.length;i++){
            if(degree[i] > 2){
                System.out.println(-1);
                return;
            }
            if(degree[i] == 1){
                begin = i;
            }
        }
        boolean[]used = new boolean[n];
        int[]p = new int[]{5,2,11,2};
        int idx = 0;
        HashMap<pair,Integer>map = new HashMap<>();
        while (!used[begin]){
            used[begin] = true;
            for(int next : grid[begin]){
                if(used[next])
                    continue;
                map.put(new pair(Math.min(begin,next),Math.max(begin,next)),p[idx % 4]);
                idx++;
                begin = next;
            }
        }
        StringBuilder bd = new StringBuilder();
        for(pair pp : list){
            bd.append(map.get(pp) + " ");
        }
        System.out.println(bd.toString().trim());
    }

    public static void main(String[] args) {
        int n = sc.nextInt();
        for(int i = 0;i < n;i++){
            solve();
        }
    }

    
}
