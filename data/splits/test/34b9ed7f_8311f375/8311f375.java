import java.io.*;
import java.util.*;

public class q3 {
    public static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
//    public static long mod = 1000000007;

    public static void solve() throws Exception {
        String[] parts = br.readLine().split(" ");
        int n = Integer.parseInt(parts[0]);


        int[] arr = new int[n + 1];
        boolean[] vis = new boolean[n + 1];

        for(int i = 1;i <= n;i++){
            if(vis[i]) continue;

            ArrayList<Integer> list = new ArrayList<>();
            while(true){
                System.out.println("? " + i);
                int val = Integer.parseInt(br.readLine());
                vis[val] = true;
                list.add(val);

                if(list.size() > 1 && val == list.get(0)) break;
            }

            for(int j = 0;j < list.size() - 1;j++) arr[list.get(j)] = list.get(j + 1);
        }

        System.out.print("! ");
        StringBuilder ans = new StringBuilder();
        for(int i = 1;i <= n;i++) ans.append(arr[i]).append(" ");
        System.out.println(ans);


    }

    public static void main(String[] args) throws Exception {
        int tests = Integer.parseInt(br.readLine());
        for (int test = 1; test <= tests; test++) {
            solve();
        }
    }

}
