import java.io.*;
import java.util.Arrays;

public class C {
    static StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in)));
    static PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
    private static int nextInt() {
        try {
            in.nextToken();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return (int) in.nval;
    }
    
    private static String next(){
        try {
            in.nextToken();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return in.sval;
    }

    public static void main(String[] args) {
        int t = nextInt();
        for (int c = 0; c < t; c++) {
            int n = nextInt();
            String[] info = new String[n];
            for (int i = 0; i < n; i++) {
                info[i] = next();
            }
            int ans = 0;
            for (int i = 0; i < 5; i++) {
                char cur = (char) (i+'a');
                int[] map = new int[n];
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < info[j].length(); k++) {
                        if(info[j].charAt(k)==cur)map[j]++;
                        else map[j]--;
                    }
                }
                Arrays.sort(map);
                int sum = 0;
                for(int j = n-1; j >= 0; j--) {
                    sum += map[j];
                    if(sum<=0){
                        break;
                    }
                    ans = Math.max(ans,n-j);
                }
            }
            System.out.println(ans);
        }
    }
}
