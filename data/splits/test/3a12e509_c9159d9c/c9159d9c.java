import java.util.*;

public class SolutionB {

    

    

    

    

    

    


    public static void main(String[] args){
        Scanner s = new Scanner(System.in);
        
        int t = s.nextInt();

        for(int i =0;i<t;i++) {
            int n = s.nextInt();
            int arr [] = new int[n];
            ArrayList<Integer> blue = new ArrayList<>();
            ArrayList<Integer> red = new ArrayList<>();

            for(int j=0;j<n;j++){
                int num = s.nextInt();
                arr[j]=num;
            }
            String color = s.next();

            for(int j=0;j<n;j++){
                if(color.charAt(j)=='B'){
                    blue.add(arr[j]);
                }
                else{
                    red.add(arr[j]);
                }
            }
            Collections.sort(blue);
            String ans = "YES";
            int counter = 0;

            for(int j=0;j<blue.size();j++){
                int current = blue.get(j);
                if (current<1){
                    ans="NO";
                    break;
                }
                if(current>counter){
                    counter++;
                }
                else{
                    ans="NO";
                    break;
                }

            }

            if(ans=="NO"){
                System.out.println(ans);
            }
            else{
                int tempCounter = n+1;
                Collections.sort(red);
                for(int j=red.size()-1;j>=0;j--){
                    int  current  = red.get(j);
                    if(current>=tempCounter){
                        ans="NO";
                        break;
                    }
                    else{
                        tempCounter--;
                    }
                }
                if(tempCounter-counter!=1)
                    System.out.println("NO");
                else
                    System.out.println(ans);
            }

        }

        return;
    }
}
