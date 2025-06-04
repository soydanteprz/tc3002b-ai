import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

public class Practice {
    static HashMap<String, Integer> map = new HashMap<>();
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        while (t-->0) {
            int n = sc.nextInt();
            int[][] occurances = new int[5][n];
            for(int i=0;i<n;i++){
                String s = sc.next();
                int[] count = new int[5];
                int len = s.length();
                for(int j=0;j<s.length();j++){
                    count[s.charAt(j)-'a']++;
                }
                for(int j=0;j<5;j++){
                    occurances[j][i] = count[j] - (len-count[j]);
                }
            }
            int ans = 0;
            for(int i=0;i<5;i++){
                Arrays.sort(occurances[i]);
                int tmpAns = 0; int tmpSum=0;
                for(int j=n-1;j>=0;j--){
                    tmpSum+=occurances[i][j];
                    if(tmpSum>0) tmpAns++;
                    else break;
                }
                ans = Math.max(ans, tmpAns);
            }
            System.out.println(ans);
        }
    }
}
