import java.io.PrintWriter;
import java.util.*;

public class D {
    //    MUST SEE BEFORE SUBMISSION
//    check whether int part would overflow or not, especially when it is a * b!!!!

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = sc.nextInt();
//        int t = 1;
        for (int i = 0; i < t; i++) {
            solve(sc, pw);
        }
        pw.close();
    }

    static void solve(Scanner in, PrintWriter out){
        int n = in.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = in.nextInt();
        }
        Set<Integer> set = new HashSet<>();
        for(int x : arr) {
            if (x == 0){
                out.println("YES");
                return;
            }else if (set.contains(x) || set.contains(-x)){
                out.println("YES");
                return;
            }
            set.add(x);
        }
        for (int i = 0; i < n; i++) {
            if (dfs(0, i, arr, 0)){
                out.println("YES");
                return;
            }
        }
        out.println("NO");
    }
    static boolean dfs(int idx, int need, int[] arr, int cur){
        if (cur == arr[need]) return true;
        if (idx == arr.length) return false;
        if (idx == need) return dfs(idx + 1, need, arr, cur);
        return dfs(idx + 1, need, arr, cur + arr[idx]) |
                dfs(idx + 1, need, arr, cur - arr[idx]) |
                dfs(idx + 1, need, arr, cur);
    }
}
